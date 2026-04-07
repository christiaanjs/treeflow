"""
Coalescent Point Process (CPP) backward algorithm for sampling BD trees.

Reference: Stadler (2011) J. Theor. Biol.; Lambert & Stadler (2013).

The CPP converts tree simulation into iid sampling from a 1-D distribution
with a closed-form inverse CDF, avoiding biased forward simulation.

Topology distribution
---------------------
``build_random_topologies`` samples uniformly over *ranked labeled tree
topologies* (labeled histories).  At each coalescence step j the two merging
lineages are chosen uniformly at random from the j+1 active lineages, so
every sequence of coalescence decisions is equally probable.  The resulting
distribution is therefore uniform over all ranked labeled topologies (the
set of labeled binary trees together with a total order on the internal
nodes consistent with the parent-child relation).
"""

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.internal import samplers

from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology


def sample_origin_age(n_taxa, lambda_, mu, rho, n_samples, seed, grid_size=1000):
    """
    Stage 1: Sample origin age T via grid-based inverse CDF.

    The marginal density of T is proportional to P(n|T, lambda, mu, rho), which
    has a closed form via the zero-inflated geometric (Stadler 2009).

    Parameters
    ----------
    n_taxa : int
        Number of sampled tips.
    lambda_, mu, rho : tf.Tensor
        BD speciation/extinction rates and sampling probability, shape [*batch_shape].
    n_samples : int
        Number of T values to draw.
    seed : seed
        TFP-compatible stateless seed.
    grid_size : int
        Number of grid points for numerical CDF inversion.

    Returns
    -------
    tf.Tensor
        Origin ages of shape [n_samples, *batch_shape].
    """
    dtype = lambda_.dtype
    eps = tf.cast(1e-8, dtype)
    n_taxa_f = tf.cast(n_taxa, dtype)

    r = lambda_ - mu  # [*batch_shape]; positive for valid BD params

    # Conservative T_max: use the minimum |r| across the batch so the grid
    # covers all batch elements.
    r_safe_scalar = tf.maximum(
        tf.reduce_min(tf.abs(r)), tf.cast(1e-2, dtype)
    )
    T_max = (
        tf.math.log(n_taxa_f + tf.cast(2.0, dtype)) + tf.cast(10.0, dtype)
    ) / r_safe_scalar
    T_max = tf.maximum(T_max, tf.cast(10.0, dtype))

    # Uniform grid [eps, T_max] with grid_size points
    grid = tf.linspace(eps, T_max, grid_size)  # [grid_size]

    # Batch info
    batch_shape = tf.shape(lambda_)  # 1-D int32 tensor, shape [ndim]
    batch_ndim = tf.size(batch_shape)  # scalar int32

    # Expand params to [*batch_shape, 1] for broadcasting against grid
    lambda_e = tf.expand_dims(lambda_, -1)
    mu_e = tf.expand_dims(mu, -1)
    rho_e = tf.expand_dims(rho, -1)
    r_e = lambda_e - mu_e
    r_e_safe = tf.maximum(tf.abs(r_e), eps)

    # Expand grid to [1,...,1, grid_size] with batch_ndim leading singleton dims
    ones_batch = tf.ones([batch_ndim], dtype=tf.int32)
    grid_e = tf.reshape(grid, tf.concat([ones_batch, tf.constant([grid_size])], 0))

    # q(T) = rho*lambda + (lambda*(1-rho) - mu)*exp(-r*T): [*batch_shape, grid_size]
    q_grid = (
        rho_e * lambda_e
        + (lambda_e * (tf.cast(1.0, dtype) - rho_e) - mu_e)
        * tf.math.exp(-r_e_safe * grid_e)
    )
    q_grid_safe = tf.maximum(q_grid, eps)

    # eta(T) = r*exp(-r*T) / q(T): proportion that has exactly 1 sampled descendant
    eta = r_e_safe * tf.math.exp(-r_e_safe * grid_e) / q_grid_safe
    eta = tf.clip_by_value(eta, tf.cast(0.0, dtype), tf.cast(1.0, dtype))

    # 1 - p0(T) = rho*r / q(T): probability of having >= 1 sampled descendant
    one_minus_p0 = rho_e * r_e_safe / q_grid_safe
    one_minus_p0 = tf.maximum(one_minus_p0, tf.cast(0.0, dtype))

    # P(n|T) = (1-p0) * eta * (1-eta)^(n-1): [*batch_shape, grid_size]
    n_minus_1 = tf.cast(n_taxa - 1, dtype)
    one_minus_eta = tf.maximum(tf.cast(1.0, dtype) - eta, eps)
    pdf = one_minus_p0 * eta * tf.math.pow(one_minus_eta, n_minus_1)

    # Cumulative trapezoid on the uniform grid (step dt)
    dt = (T_max - eps) / tf.cast(grid_size - 1, dtype)
    areas = (
        tf.cast(0.5, dtype) * (pdf[..., :-1] + pdf[..., 1:]) * dt
    )  # [*batch_shape, grid_size-1]
    cdf = tf.cumsum(areas, axis=-1)  # [*batch_shape, grid_size-1]

    # Normalise
    cdf_total = tf.maximum(cdf[..., -1:], eps)
    cdf = cdf / cdf_total  # [*batch_shape, grid_size-1]

    # The CDF values correspond to grid[1:]
    grid_cdf = grid[1:]  # [grid_size-1]
    n_cdf = grid_size - 1

    # Draw uniform samples: [n_samples, *batch_shape]
    u_shape = tf.concat([tf.constant([n_samples]), batch_shape], 0)
    u = tf.random.stateless_uniform(
        u_shape, seed, minval=eps, maxval=tf.cast(1.0, dtype) - eps, dtype=dtype
    )

    # Expand CDF for searchsorted: [n_samples, *batch_shape, grid_size-1]
    cdf_expanded = tf.broadcast_to(
        tf.expand_dims(cdf, 0),
        tf.concat([tf.constant([n_samples]), batch_shape, tf.constant([n_cdf])], 0),
    )
    u_col = tf.expand_dims(u, -1)  # [n_samples, *batch_shape, 1]

    # Inverse CDF: find grid index via searchsorted
    idx = tf.searchsorted(cdf_expanded, u_col, side="right")  # [n_samples, *batch_shape, 1]
    idx = tf.squeeze(idx, axis=-1)  # [n_samples, *batch_shape]
    idx = tf.clip_by_value(idx, 0, n_cdf - 1)

    T = tf.gather(grid_cdf, idx)  # [n_samples, *batch_shape]
    return T


def sample_speciation_times(T, n_taxa, lambda_, mu, rho, seed):
    """
    Stage 2: Sample n_taxa-1 iid speciation times via closed-form inverse CDF.

    The antiderivative of the pulled speciation rate lambda_p(tau) is:
        Lambda(tau) = rho * r / q(tau)
    where q(tau) = rho*lambda + (lambda*(1-rho) - mu)*exp(-r*tau).

    The inverse CDF from u in [0,1] is:
        tau = -log((rho*r/C - rho*lambda) / (lambda*(1-rho) - mu)) / r
    where C = Lambda(0) + u * (Lambda(T) - Lambda(0)).

    Parameters
    ----------
    T : tf.Tensor
        Origin ages, shape [n_samples, *batch_shape].
    n_taxa : int
        Number of tips.
    lambda_, mu, rho : tf.Tensor
        BD parameters, shape [*batch_shape].
    seed : seed
        TFP-compatible stateless seed.

    Returns
    -------
    tf.Tensor
        Speciation times sorted ascending (youngest to oldest),
        shape [n_samples, *batch_shape, n_taxa-1].
    """
    dtype = lambda_.dtype
    eps = tf.cast(1e-8, dtype)

    r = lambda_ - mu  # [*batch_shape]
    r_safe = tf.where(tf.abs(r) < eps, tf.ones_like(r) * eps, r)

    # q(T): params [*batch_shape] broadcast with T [n_samples, *batch_shape]
    q_T = (
        rho * lambda_
        + (lambda_ * (tf.cast(1.0, dtype) - rho) - mu) * tf.math.exp(-r_safe * T)
    )
    q_T_safe = tf.maximum(q_T, eps)

    # Lambda(T) = rho * r / q(T): [n_samples, *batch_shape]
    Lambda_T = rho * r_safe / q_T_safe
    # Lambda(0) = rho (same for all T)
    Lambda_0 = rho  # [*batch_shape]

    # Draw uniform samples: [n_samples, *batch_shape, n_taxa-1]
    n_samples_t = tf.shape(T)[0]  # scalar TF int
    batch_shape = tf.shape(lambda_)
    u_shape = tf.concat(
        [tf.expand_dims(n_samples_t, 0), batch_shape, tf.constant([n_taxa - 1])], 0
    )
    u = tf.random.stateless_uniform(
        u_shape, seed, minval=eps, maxval=tf.cast(1.0, dtype) - eps, dtype=dtype
    )

    # C = Lambda(0) + u * (Lambda(T) - Lambda(0)): [n_samples, *batch_shape, n_taxa-1]
    Lambda_T_e = tf.expand_dims(Lambda_T, -1)     # [n_samples, *batch_shape, 1]
    Lambda_0_e = tf.expand_dims(Lambda_0, -1)     # [*batch_shape, 1]
    C = Lambda_0_e + u * (Lambda_T_e - Lambda_0_e)  # [n_samples, *batch_shape, n_taxa-1]
    C_safe = tf.maximum(C, eps)

    # Expand params to [1, *batch_shape, 1] for broadcasting with C
    r_ee = tf.expand_dims(tf.expand_dims(r_safe, 0), -1)
    rho_ee = tf.expand_dims(tf.expand_dims(rho, 0), -1)
    lambda_ee = tf.expand_dims(tf.expand_dims(lambda_, 0), -1)
    mu_ee = tf.expand_dims(tf.expand_dims(mu, 0), -1)

    # Numerator of log arg: rho*r/C - rho*lambda: [n_samples, *batch_shape, n_taxa-1]
    num = rho_ee * r_ee / C_safe - rho_ee * lambda_ee

    # Denominator: lambda*(1-rho) - mu: [1, *batch_shape, 1]
    den = lambda_ee * (tf.cast(1.0, dtype) - rho_ee) - mu_ee
    den_safe = tf.where(
        tf.abs(den) < eps,
        tf.math.sign(den + eps) * eps,
        den,
    )

    # Both num and den have the same sign for valid (u, T), so log_arg > 0
    log_arg = num / den_safe
    log_arg_safe = tf.maximum(log_arg, eps)

    tau = -tf.math.log(log_arg_safe) / r_ee  # [n_samples, *batch_shape, n_taxa-1]

    # Clip to [0, T] for numerical safety
    T_e = tf.expand_dims(T, -1)  # [n_samples, *batch_shape, 1]
    tau = tf.clip_by_value(tau, tf.cast(0.0, dtype), T_e)

    # Sort ascending (youngest first = smallest height first)
    tau = tf.sort(tau, axis=-1)
    return tau


def build_random_topologies(n_total, n_taxa, seed):
    """
    Stage 3: Build n_total random labeled ranked tree topologies via pure TF.

    Uses only TF stateless random ops and tensor operations — no NumPy, no
    .numpy() calls — so it is fully compatible with tf.function tracing.

    The topology distribution is uniform over ranked labeled topologies
    (labeled histories): at each step j the pair of merging lineages is chosen
    uniformly from the C(j+1, 2) available pairs, so all ranked topologies are
    equally probable.

    ``preorder_indices`` is constructed directly from the creation order rather
    than via DFS: internal nodes are created in postorder, so their reverse
    (root first) is a valid preorder.  The same order holds for every sample.

    Parameters
    ----------
    n_total : int
        Number of independent topologies to generate (Python int).
    n_taxa : int
        Number of leaf taxa (Python int).
    seed : seed
        TFP-compatible stateless seed.

    Returns
    -------
    TensorflowTreeTopology
        parent_indices:    [n_total, 2*n_taxa-2]
        child_indices:     [n_total, 2*n_taxa-1, 2]
        preorder_indices:  [n_total, 2*n_taxa-1]
    """
    n = n_taxa
    node_count = 2 * n - 1

    if n_total == 0:
        return TensorflowTreeTopology(
            parent_indices=tf.zeros([0, 2 * n - 2], dtype=tf.int32),
            child_indices=tf.fill([0, node_count, 2], tf.constant(-1, dtype=tf.int32)),
            preorder_indices=tf.zeros([0, node_count], dtype=tf.int32),
        )

    # Split a seed per coalescence step
    if n > 1:
        step_seeds = samplers.split_seed(seed, n=n - 1)
    else:
        step_seeds = []

    s_idx = tf.range(n_total, dtype=tf.int32)

    # --- Initialise state tensors (all int32) ---
    parent_indices = tf.fill([n_total, 2 * n - 2], tf.constant(-1, dtype=tf.int32))
    child_indices = tf.fill([n_total, node_count, 2], tf.constant(-1, dtype=tf.int32))
    # active[s, :n_active] contains currently active lineages; initialised to leaf indices
    active = tf.tile(
        tf.expand_dims(tf.range(n, dtype=tf.int32), 0), [n_total, 1]
    )  # [n_total, n]

    for j in range(n - 1):
        n_active = n - j       # Python int: number of active lineages this step
        internal_node = n + j  # Python int: index of the new internal node

        seed_a, seed_b = samplers.split_seed(step_seeds[j], n=2)

        # Sample two distinct positions in the active list
        pos_a = tf.random.stateless_uniform(
            [n_total], seed_a, minval=0, maxval=n_active, dtype=tf.int32
        )
        if n_active > 1:
            pos_b_raw = tf.random.stateless_uniform(
                [n_total], seed_b, minval=0, maxval=n_active - 1, dtype=tf.int32
            )
            pos_b = tf.where(pos_b_raw >= pos_a, pos_b_raw + 1, pos_b_raw)
        else:
            pos_b = tf.zeros([n_total], dtype=tf.int32)

        # Gather the two coalescing lineages
        ca = tf.gather_nd(active, tf.stack([s_idx, pos_a], axis=1))  # [n_total]
        cb = tf.gather_nd(active, tf.stack([s_idx, pos_b], axis=1))

        # --- Update parent_indices ---
        int_fill = tf.fill([n_total], internal_node)
        parent_indices = tf.tensor_scatter_nd_update(
            parent_indices, tf.stack([s_idx, ca], axis=1), int_fill
        )
        parent_indices = tf.tensor_scatter_nd_update(
            parent_indices, tf.stack([s_idx, cb], axis=1), int_fill
        )

        # --- Update child_indices (store children sorted: min first) ---
        child0 = tf.minimum(ca, cb)
        child1 = tf.maximum(ca, cb)
        int_node_fill = tf.fill([n_total], internal_node)
        zero_fill = tf.zeros([n_total], dtype=tf.int32)
        one_fill = tf.ones([n_total], dtype=tf.int32)
        child_indices = tf.tensor_scatter_nd_update(
            child_indices,
            tf.stack([s_idx, int_node_fill, zero_fill], axis=1),
            child0,
        )
        child_indices = tf.tensor_scatter_nd_update(
            child_indices,
            tf.stack([s_idx, int_node_fill, one_fill], axis=1),
            child1,
        )

        # --- Update active list ---
        # Remove ca (at pos_a) and cb (at pos_b), add internal_node.
        # Strategy: keep n_active-1 elements in positions 0..n_active-2.
        #
        # Three cases based on whether pos_a or pos_b is the last active slot:
        #   Case 1 (pos_a == last): write internal_node at pos_b; drop last slot.
        #   Case 2 (pos_b == last): write internal_node at pos_a; drop last slot.
        #   Case 3 (normal):        write internal_node at pos_a;
        #                           fill pos_b with the last active element.
        last_pos_fill = tf.fill([n_total], n_active - 1)
        last_elems = tf.gather_nd(active, tf.stack([s_idx, last_pos_fill], axis=1))

        is_pa_last = tf.equal(pos_a, n_active - 1)
        is_pb_last = tf.equal(pos_b, n_active - 1)
        is_normal = tf.logical_not(tf.logical_or(is_pa_last, is_pb_last))

        # Write 1: place internal_node at pos_b (case 1) or pos_a (cases 2 & 3)
        write1_pos = tf.where(is_pa_last, pos_b, pos_a)
        active = tf.tensor_scatter_nd_update(
            active,
            tf.stack([s_idx, write1_pos], axis=1),
            int_fill,
        )

        # Write 2: fill pos_b appropriately
        #   case 1: write internal_node back (no-op — already written to pos_b)
        #   case 2: write cb back (no-op — pos_b==last, not used after decrement)
        #   case 3: write last_elem into pos_b to fill the gap
        value2 = tf.where(
            is_pa_last,
            int_fill,
            tf.where(is_pb_last, cb, last_elems),
        )
        active = tf.tensor_scatter_nd_update(
            active,
            tf.stack([s_idx, pos_b], axis=1),
            value2,
        )

    # Internal nodes are created in postorder (n_taxa, n_taxa+1, ..., root) because
    # each new node is added only after both its children are active lineages.
    # Reversing the creation order gives a valid preorder (root first, every parent
    # before its children).  Leaves are appended last; since every leaf's parent is
    # an internal node, the parent-before-child invariant still holds.
    # This preorder is the same for every sample in the batch, so it is tiled.
    internal_preorder = tf.range(node_count - 1, n - 1, -1, dtype=tf.int32)  # [n-1]
    leaf_indices = tf.range(n, dtype=tf.int32)                                 # [n]
    row = tf.concat([internal_preorder, leaf_indices], axis=0)                 # [node_count]
    preorder_indices = tf.tile(tf.expand_dims(row, 0), [n_total, 1])          # [n_total, node_count]

    return TensorflowTreeTopology(
        parent_indices=parent_indices,
        child_indices=child_indices,
        preorder_indices=preorder_indices,
    )


def sample_bd_tree(
    n_taxa,
    lambda_,
    mu,
    rho,
    n_samples,
    seed,
    fixed_topology=None,
    fixed_origin_age=None,
):
    """
    Orchestrator: sample n_samples BD trees from the CPP backward algorithm.

    Parameters
    ----------
    n_taxa : int
        Number of leaf taxa.
    lambda_, mu, rho : tf.Tensor
        BD parameters, shape [*batch_shape].
    n_samples : int
        Number of independent samples (the n from _sample_n).
    seed : seed
        TFP-compatible stateless seed (None is allowed).
    fixed_topology : TensorflowTreeTopology, optional
        Skip Stage 3 and broadcast this topology.
    fixed_origin_age : tf.Tensor, optional
        Skip Stage 1 and use this origin age, shape [n_samples, *batch_shape].

    Returns
    -------
    TensorflowRootedTree
        Sampled trees; each component has shape [n_samples, *batch_shape, ...].
    """
    seed = samplers.sanitize_seed(seed)
    seed1, seed2, seed3 = samplers.split_seed(seed, n=3)

    dtype = lambda_.dtype
    batch_shape = tf.shape(lambda_)  # 1-D int32 tensor

    # --- Stage 1: origin age ---
    if fixed_origin_age is None:
        T = sample_origin_age(n_taxa, lambda_, mu, rho, n_samples, seed1)
    else:
        T = fixed_origin_age  # [n_samples, *batch_shape]

    # --- Stage 2: speciation times (internal node heights) ---
    heights = sample_speciation_times(T, n_taxa, lambda_, mu, rho, seed2)
    # heights: [n_samples, *batch_shape, n_taxa-1]

    # --- Stage 3: random topology ---
    if fixed_topology is None:
        # Prefer static shape so this works inside tf.function tracing too
        shape_list = lambda_.shape.as_list()
        if any(d is None for d in shape_list):
            n_batch = int(tf.reduce_prod(tf.shape(lambda_)))
        else:
            n_batch = int(np.prod(shape_list)) if shape_list else 1
        n_total = n_samples * max(n_batch, 1)

        flat_topology = build_random_topologies(n_total, n_taxa, seed3)

        # Reshape [n_total, ...] → [n_samples, *batch_shape, ...]
        def _reshape(t, *event_dims):
            target = tf.concat(
                [tf.constant([n_samples]), batch_shape, tf.constant(list(event_dims))],
                0,
            )
            return tf.reshape(t, target)

        topology = TensorflowTreeTopology(
            parent_indices=_reshape(flat_topology.parent_indices, 2 * n_taxa - 2),
            child_indices=_reshape(flat_topology.child_indices, 2 * n_taxa - 1, 2),
            preorder_indices=_reshape(flat_topology.preorder_indices, 2 * n_taxa - 1),
        )
    else:
        topology = fixed_topology

    # Sampling times: all zero (contemporaneous sampling)
    sampling_times = tf.zeros(
        tf.concat([tf.constant([n_samples]), batch_shape, tf.constant([n_taxa])], 0),
        dtype=dtype,
    )

    return TensorflowRootedTree(
        node_heights=heights,
        sampling_times=sampling_times,
        topology=topology,
    )


__all__ = [
    "sample_origin_age",
    "sample_speciation_times",
    "build_random_topologies",
    "sample_bd_tree",
]
