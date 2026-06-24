"""Generalized pruning over the full subsplit DAG.

The conditional clade distribution defines, for every clade ``c``, a categorical
over its subsplits with weights ``w_{(c,s)}`` (a softmax within each parent-clade
group). Instead of *sampling* a tree and pruning it, **generalized pruning** runs
Felsenstein's recursion over the whole subsplit DAG at once, marginalising the
topology by a sum-product::

    Pi[c] = sum over subsplits s = (X, Y) of c   w_{(c,s)} * (P_X . Pi[X]) (x) (P_Y . Pi[Y])

with ``Pi[leaf i]`` the observed one-hot, ``P_c`` the transition matrix on the
edge above clade ``c``, and ``(x)`` the elementwise product. At the root,
``sum_state freq * Pi[root]`` is the **per-site tree-marginal likelihood**
``E_{q(T)}[L_s(T)]`` -- exactly, because the CCD factorises so each clade's
sub-DAG expectation is reused by every tree that contains the clade. (See
:func:`relaxed_partials_sequential` for the inductive argument.)

This gives a *deterministic, exact-autodiff* gradient of the (per-site)
tree-marginal likelihood w.r.t. the clade weights -- no sampling and no
straight-through bias -- in time linear in the number of subsplits (the DAG size)
rather than the ``(2n-3)!!`` topologies. It is the tractable analogue, for an
enumerable DAG, of the straight-through likelihood.

**Pluggable decision weights.** The weights ``w`` need not be the exact marginal
conditional probabilities. Any per-parent-group weighting can be dropped in (see
the ``*_weights`` samplers): the *exact* marginal gives the per-site tree-marginal
above, while a **(Gumbel-)softmax sample** relaxes a single drawn tree. The key
fact -- exactly your intuition -- is that when the weights are a **hard** one-hot
per clade, the sum-product collapses onto a single tree and the objective becomes
the proper *across-sites* likelihood ``sum_s log L_s(T) = log L(T)`` (no Jensen
gap). So a straight-through weighting (hard one-hot forward, relaxed backward)
turns generalized pruning into a Gumbel-softmax / straight-through topology
estimator on the **correct** joint objective, where a single Gumbel-max draw per
clade is an exact CCD tree sample and supplies *every* clade's partial at once.

Two equivalent passes are provided over the *static* DAG (so both unroll cleanly
under ``tf.function``):

* :func:`relaxed_partials_sequential` -- an unrolled loop over clades in
  child-before-parent order. Simplest: each clade reads partials that are already
  final, no masking, and it works with arbitrary leading (site/batch) dimensions.
* :func:`relaxed_partials_vectorized` -- a level-based pass (one iteration per
  clade *size*) that computes every subsplit's contribution in a single batched
  ``einsum`` and scatters into parents with a segment sum. Fewer, larger ops --
  the shape that maps to a native kernel -- at the cost of recomputing masked
  contributions. Assumes a single leading site axis.
"""

from __future__ import annotations

import typing as tp

import numpy as np
import tensorflow as tf

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.conditional_clade.clade import clade_size
from treeflow.conditional_clade.distribution import segment_log_softmax


# ----------------------------------------------------------------------------
# Decision-weight samplers
#
# Each maps clade logits to subsplit weights ``w`` (flat order, summing to one
# within every parent-clade segment), to be fed to the DAG sum-product. They share
# the signature ``(logits, segment_ids, num_segments, temperature, seed)`` so they
# are interchangeable as the ``weight_fn`` of
# :func:`relaxed_log_likelihood_from_distribution`.
# ----------------------------------------------------------------------------
def _gumbel_like(logits: tf.Tensor, seed=None) -> tf.Tensor:
    u = tf.random.uniform(
        tf.shape(logits), minval=1e-20, maxval=1.0, dtype=logits.dtype, seed=seed
    )
    return -tf.math.log(-tf.math.log(u))


def _segment_one_hot_argmax(
    values: tf.Tensor, segment_ids: tf.Tensor, num_segments: int
) -> tf.Tensor:
    """One-hot of the argmax within each segment (lowest index breaks ties)."""
    seg_max = tf.math.unsorted_segment_max(values, segment_ids, num_segments)
    is_max = values >= tf.gather(seg_max, segment_ids)
    idx = tf.range(tf.size(values))
    sentinel = tf.fill(tf.shape(idx), tf.size(values))
    masked_idx = tf.where(is_max, idx, sentinel)
    seg_min_idx = tf.math.unsorted_segment_min(masked_idx, segment_ids, num_segments)
    chosen = tf.equal(idx, tf.gather(seg_min_idx, segment_ids))
    return tf.cast(chosen, values.dtype)


def exact_weights(
    logits, segment_ids, num_segments, temperature=1.0, seed=None
) -> tf.Tensor:
    """The exact marginal conditional probabilities (no sampling)."""
    return tf.exp(segment_log_softmax(logits, segment_ids, num_segments))


def gumbel_softmax_weights(
    logits, segment_ids, num_segments, temperature=1.0, seed=None
) -> tf.Tensor:
    """A relaxed (Gumbel-softmax) tree sample: ``softmax((logits + g) / tau)``."""
    perturbed = logits + _gumbel_like(logits, seed)
    return tf.exp(segment_log_softmax(perturbed / temperature, segment_ids, num_segments))


def straight_through_weights(
    logits, segment_ids, num_segments, temperature=1.0, seed=None, gumbel=True
) -> tf.Tensor:
    """Hard one-hot per clade (forward), relaxed softmax (backward).

    With ``gumbel=True`` the hard choice is a Gumbel-max draw -- an exact CCD tree
    sample -- so the forward sum-product collapses to that tree's *across-sites*
    likelihood; the backward pass flows through the (Gumbel-)softmax. With
    ``gumbel=False`` the hard choice is the plain argmax (the mode).
    """
    perturbed = logits + _gumbel_like(logits, seed) if gumbel else logits
    soft = tf.exp(segment_log_softmax(perturbed / temperature, segment_ids, num_segments))
    hard = _segment_one_hot_argmax(perturbed, segment_ids, num_segments)
    return hard + soft - tf.stop_gradient(soft)


def gumbel_straight_through_weights(
    logits, segment_ids, num_segments, temperature=1.0, seed=None
) -> tf.Tensor:
    """Gumbel-softmax straight-through (alias of ``straight_through_weights`` with a
    Gumbel-max hard sample)."""
    return straight_through_weights(
        logits, segment_ids, num_segments, temperature, seed, gumbel=True
    )


class SubsplitDAG:
    """Static index structure for generalized pruning over a clade support.

    Rows index *clades*: leaves ``0..n-1`` (so row ``i`` is leaf ``i``), then the
    splittable clades ordered smallest-first (children before parents). Subsplits
    keep the support's flat order; ``par``/``left``/``right`` give the clade row of
    each subsplit's parent and two children.
    """

    def __init__(self, support):
        n = support.taxon_count
        self.support = support
        self.taxon_count = n

        clade_row: tp.Dict[int, int] = {1 << i: i for i in range(n)}
        # Splittable clades smallest-first (support orders them largest-first).
        internal = list(reversed(support.parent_clades))
        for k, clade in enumerate(internal):
            clade_row[clade] = n + k
        self.internal_clades = internal
        self.clade_row = clade_row
        self.num_clades = n + len(internal)
        self.root_index = clade_row[support.root_clade]

        flat_subsplits = support.flat_subsplits
        flat_parents = support.flat_parents
        self.subsplit_count = len(flat_subsplits)
        self.par = np.array([clade_row[p] for p in flat_parents], dtype=np.int32)
        self.left = np.array([clade_row[s.child1] for s in flat_subsplits], dtype=np.int32)
        self.right = np.array([clade_row[s.child2] for s in flat_subsplits], dtype=np.int32)

        # Sequential: for each internal clade (smallest-first) the contiguous
        # range of its subsplits in the flat order.
        self.clade_subsplit_slices: tp.List[tp.Tuple[int, int]] = []
        for clade in internal:
            pidx = support.parent_clade_index[clade]
            start = support.parent_offsets[pidx]
            count = len(support.subsplits_by_parent[pidx])
            self.clade_subsplit_slices.append((start, count))

        # Vectorized: a subsplit's level is its parent clade's size (2..n); all of
        # a clade's children are strictly smaller, hence done at earlier levels.
        par_size = np.array([clade_size(p) for p in flat_parents], dtype=np.int32)
        self.num_levels = n - 1
        self.level_mask = np.stack(
            [par_size == lvl + 2 for lvl in range(self.num_levels)]
        )  # [num_levels, subsplit_count] bool


def build_tip_partials(
    dag: SubsplitDAG, sequences_onehot: tf.Tensor
) -> tf.Tensor:
    """Stack leaf one-hots into ``[num_clades, ..., state]`` (internal rows zero)."""
    n = dag.taxon_count
    leaves = [sequences_onehot[..., i, :] for i in range(n)]
    zero = tf.zeros_like(leaves[0])
    rows = leaves + [zero] * (dag.num_clades - n)
    return tf.stack(rows, axis=0)


def build_transition_matrices(
    dag: SubsplitDAG,
    transition_fn: tp.Callable[[int], tf.Tensor],
    dtype: tf.DType = DEFAULT_FLOAT_DTYPE_TF,
) -> tf.Tensor:
    """``[num_clades, state, state]`` edge matrices from a ``clade -> [s, s]`` fn.

    The matrix on the edge *above* each clade. The root has no edge above it and is
    never used as a child, so its row is filled with the identity.
    """
    rows: tp.List[tf.Tensor] = [None] * dag.num_clades  # type: ignore[list-item]
    state = None
    for clade, row in dag.clade_row.items():
        if row == dag.root_index:
            continue
        mat = transition_fn(clade)
        rows[row] = mat
        state = int(mat.shape[-1])
    rows[dag.root_index] = tf.eye(state, dtype=dtype)
    return tf.stack(rows, axis=0)


def relaxed_partials_sequential(
    dag: SubsplitDAG, w: tf.Tensor, P: tf.Tensor, tip_partials: tf.Tensor
) -> tf.Tensor:
    """Generalized pruning by an unrolled child-before-parent loop.

    ``Pi[c] = sum_s w_s (P_X . Pi[X]) (x) (P_Y . Pi[Y])``. Because ``Pi[X]`` is, by
    induction, ``E[`` partial below ``X ]`` and the CCD makes the two child
    subtrees conditionally independent given the subsplit, this equals
    ``E_{q(subtree|c)}[`` partial below ``c ]``; at the root it is ``E_q[`` root
    partial ``]``. Returns ``Pi`` as ``[num_clades, ..., state]``.
    """
    Pi: tp.List[tf.Tensor] = [tip_partials[i] for i in range(dag.taxon_count)]
    for ci, _clade in enumerate(dag.internal_clades):
        start, count = dag.clade_subsplit_slices[ci]
        contrib = []
        for t in range(start, start + count):
            left, right = int(dag.left[t]), int(dag.right[t])
            l = tf.linalg.matvec(P[left], Pi[left])
            r = tf.linalg.matvec(P[right], Pi[right])
            contrib.append(w[t] * (l * r))
        Pi.append(tf.add_n(contrib))
    return tf.stack(Pi, axis=0)


def relaxed_partials_vectorized(
    dag: SubsplitDAG, w: tf.Tensor, P: tf.Tensor, tip_partials: tf.Tensor
) -> tf.Tensor:
    """Generalized pruning as a level-based, batched sum-product.

    One iteration per clade *size*: gather every subsplit's children, combine in a
    single ``einsum``, mask to the current level, and segment-sum into parents.
    Equivalent to :func:`relaxed_partials_sequential` but with fewer, larger ops.
    ``tip_partials`` must have a single leading site axis: ``[num_clades, sites,
    state]``.
    """
    PL = tf.gather(P, dag.left)   # [subsplit, state, state]
    PR = tf.gather(P, dag.right)
    w_col = w[:, tf.newaxis, tf.newaxis]  # [subsplit, 1, 1]
    par = tf.constant(dag.par)
    level_mask = tf.constant(dag.level_mask)  # [num_levels, subsplit]

    Pi = tip_partials
    for lvl in range(dag.num_levels):
        l = tf.einsum("tab,tsb->tsa", PL, tf.gather(Pi, dag.left))
        r = tf.einsum("tab,tsb->tsa", PR, tf.gather(Pi, dag.right))
        contrib = w_col * (l * r)  # [subsplit, sites, state]
        mask = tf.cast(level_mask[lvl][:, tf.newaxis, tf.newaxis], contrib.dtype)
        Pi = Pi + tf.math.unsorted_segment_sum(contrib * mask, par, dag.num_clades)
    return Pi


def relaxed_log_likelihood(
    dag: SubsplitDAG,
    w: tf.Tensor,
    P: tf.Tensor,
    tip_partials: tf.Tensor,
    frequencies: tf.Tensor,
    vectorized: bool = False,
) -> tf.Tensor:
    """Per-site tree-marginal log-likelihood ``sum_s log E_q[L_s(T)]``.

    ``w`` are the clade weights (``exp`` of the conditional log-probs, summing to
    one within each parent-clade group); the gradient w.r.t. ``w`` -- hence the
    clade logits -- is exact (deterministic autodiff over the DAG).
    """
    partials_fn = (
        relaxed_partials_vectorized if vectorized else relaxed_partials_sequential
    )
    Pi = partials_fn(dag, w, P, tip_partials)
    root = Pi[dag.root_index]
    site_likelihood = tf.reduce_sum(frequencies * root, axis=-1)
    return tf.reduce_sum(tf.math.log(site_likelihood))


def relaxed_log_likelihood_from_distribution(
    q_distribution,
    P: tf.Tensor,
    sequences_onehot: tf.Tensor,
    frequencies: tf.Tensor,
    dag: tp.Optional[SubsplitDAG] = None,
    vectorized: bool = False,
    weight_fn: tp.Optional[tp.Callable] = None,
    temperature: float = 1.0,
    seed=None,
) -> tf.Tensor:
    """Generalized-pruning log-likelihood for a :class:`ConditionalCladeDistribution`.

    By default ``w`` is the exact marginal (``q_distribution.conditional_log_probs``),
    giving the deterministic per-site tree-marginal. Pass a sampler as ``weight_fn``
    -- e.g. :func:`gumbel_softmax_weights` or :func:`straight_through_weights` -- to
    relax a *single* drawn tree instead; with a hard (straight-through) ``weight_fn``
    the forward pass is one tree's across-sites likelihood. The returned scalar is
    differentiable in the clade logits.
    """
    if dag is None:
        dag = SubsplitDAG(q_distribution.support)
    if weight_fn is None:
        w = tf.exp(q_distribution.conditional_log_probs())
    else:
        logits = tf.convert_to_tensor(q_distribution.logits, q_distribution.dtype)
        segment_ids = tf.constant(q_distribution.support.segment_ids, tf.int32)
        w = weight_fn(
            logits,
            segment_ids,
            q_distribution.support.parent_clade_count,
            temperature=temperature,
            seed=seed,
        )
    tip_partials = build_tip_partials(dag, sequences_onehot)
    return relaxed_log_likelihood(
        dag, w, P, tip_partials, frequencies, vectorized=vectorized
    )


__all__ = [
    "SubsplitDAG",
    "build_tip_partials",
    "build_transition_matrices",
    "relaxed_partials_sequential",
    "relaxed_partials_vectorized",
    "relaxed_log_likelihood",
    "relaxed_log_likelihood_from_distribution",
    "exact_weights",
    "gumbel_softmax_weights",
    "straight_through_weights",
    "gumbel_straight_through_weights",
]
