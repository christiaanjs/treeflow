"""Metropolis-Hastings MCMC over rooted tree topologies.

This is the sampling-based counterpart to VBPI: instead of fitting a subsplit
Bayesian network by optimisation, it draws topologies from the posterior with a
Metropolis-Hastings random walk. Everything stays in *rooted*-tree space -- the
moves are rooted nearest-neighbour interchanges (NNI) and the target is scored
on rooted trees (see :mod:`treeflow.vbpi.likelihood`).

Two layers are provided, mirroring the structure of the HMC code
(:mod:`treeflow.vi.hmc`), which wraps TensorFlow Probability's
``TransitionKernel`` abstraction and drives it with ``sample_chain``:

* :class:`TopologyMetropolisHastings` -- a ``tfp.mcmc.TransitionKernel`` whose
  state is a rooted topology (``parent_indices``) and whose proposal is a uniform
  random rooted NNI. Because every rooted binary tree on ``n`` taxa has exactly
  ``2(n-2)`` NNI neighbours, the proposal is symmetric and the acceptance ratio
  is the plain target log-density difference. It targets an arbitrary
  ``target_log_prob_fn(parent_indices) -> scalar`` and runs eagerly.
* :func:`sample_phylogenetic_topologies` -- a Metropolis-within-Gibbs sampler for
  the *joint* phylogenetic posterior over ``(topology, branch lengths)`` under
  Jukes-Cantor, alternating an NNI topology move (carrying branch lengths through
  the relabelling) with a random-walk branch-length update. Marginalising its
  topology samples gives the posterior over topologies that the notebook compares
  against VBPI's subsplit distribution.

Rooted NNI
----------
An internal edge of a rooted tree joins a non-root internal node ``c`` to its
(internal) parent ``p``. Let ``s`` be ``c``'s sibling (``p``'s other child) and
``a``, ``b`` be ``c``'s children. The two NNI moves on that edge swap ``s`` with
``a`` or with ``b``. The reverse move recovers the original tree, so NNI is a
symmetric, irreducible move set on rooted topologies.
"""
import typing as tp

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from treeflow import DEFAULT_FLOAT_DTYPE_NP, DEFAULT_FLOAT_DTYPE_TF
from treeflow.vbpi.clade import child_indices_from_parent_indices


# ---------------------------------------------------------------------------
# Topology relabelling and NNI moves (rooted, treeflow index convention)
# ---------------------------------------------------------------------------
def canonicalize_parent_indices(
    parent_indices: np.ndarray, taxon_count: int
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Relabel internal nodes into treeflow's canonical convention.

    After an NNI swap the node labels may violate the children<parent, root-last
    convention. This recomputes a *canonical* labelling (leaves ``0..n-1`` fixed,
    internal nodes numbered in postorder, root ``2n-2`` last) and returns the new
    ``parent_indices`` together with the ``old_id -> new_id`` map (length
    ``2n-1``), which callers use to carry per-node quantities (e.g. branch
    lengths) through the relabelling.

    The labelling is structurally canonical: at each internal node the child
    whose subtree contains the smaller minimum leaf is visited first, so the same
    tree always maps to the same array regardless of the input labelling.
    """
    node_count = 2 * taxon_count - 1
    parent_indices = np.asarray(parent_indices)
    root = node_count - 1
    children = child_indices_from_parent_indices(parent_indices, node_count)

    # Pass 1: minimum leaf in each node's subtree (postorder over ascending... the
    # input labels need not be canonical, so traverse explicitly).
    min_leaf = np.full(node_count, node_count, dtype=np.int64)
    stack1: tp.List[tp.Tuple[int, int]] = [(root, 0)]
    while stack1:
        node, phase = stack1.pop()
        if node < taxon_count:
            min_leaf[node] = node
            continue
        if phase == 0:
            stack1.append((node, 1))
            for c in children[node]:
                stack1.append((c, 0))
        else:
            min_leaf[node] = min(min_leaf[c] for c in children[node])

    old_to_new = np.full(node_count, -1, dtype=np.int64)
    for leaf in range(taxon_count):
        old_to_new[leaf] = leaf
    counter = taxon_count
    # Pass 2: postorder assigning internal ids, visiting the smaller-min-leaf
    # child first (push larger first so it is popped/assigned later).
    stack: tp.List[tp.Tuple[int, int]] = [(root, 0)]
    while stack:
        node, phase = stack.pop()
        if node < taxon_count:
            continue
        if phase == 0:
            stack.append((node, 1))
            ordered = sorted(children[node], key=lambda c: min_leaf[c], reverse=True)
            for c in ordered:
                stack.append((c, 0))
        else:
            old_to_new[node] = counter
            counter += 1
    assert old_to_new[root] == node_count - 1

    new_parent = np.full(node_count - 1, -1, dtype=np.int32)
    for node in range(node_count):
        if node == root:
            continue
        new_parent[old_to_new[node]] = old_to_new[parent_indices[node]]
    return new_parent, old_to_new


def _internal_edge_nodes(taxon_count: int) -> tp.List[int]:
    """Non-root internal node ids (each defines one internal edge to its parent)."""
    return list(range(taxon_count, 2 * taxon_count - 2))


def num_nni_neighbours(taxon_count: int) -> int:
    """Number of NNI neighbours of any rooted binary tree: ``2(n-2)``."""
    return 2 * max(taxon_count - 2, 0)


def _apply_nni(
    parent_indices: np.ndarray, c: int, swap_b: bool
) -> np.ndarray:
    """Return the (non-canonical) parent array after one NNI on edge above ``c``.

    ``c`` is a non-root internal node; ``swap_b`` selects which child of ``c`` is
    exchanged with ``c``'s sibling.
    """
    parent = np.asarray(parent_indices).copy()
    node_count = parent.shape[-1] + 1
    children = child_indices_from_parent_indices(parent, node_count)
    p = int(parent[c])
    siblings = [x for x in children[p] if x != c]
    s = siblings[0]
    a, b = children[c]
    moved_child = b if swap_b else a
    # Swap subtree ``moved_child`` (under c) with ``s`` (under p).
    parent[moved_child] = p
    parent[s] = c
    return parent


def rooted_nni_neighbours(
    parent_indices: np.ndarray, taxon_count: int
) -> tp.List[np.ndarray]:
    """All ``2(n-2)`` canonicalised NNI neighbours of a rooted topology."""
    neighbours = []
    for c in _internal_edge_nodes(taxon_count):
        for swap_b in (False, True):
            moved = _apply_nni(parent_indices, c, swap_b)
            new_parent, _ = canonicalize_parent_indices(moved, taxon_count)
            neighbours.append(new_parent)
    return neighbours


def propose_nni(
    parent_indices: np.ndarray,
    taxon_count: int,
    rng: np.random.Generator,
    branch_lengths: tp.Optional[np.ndarray] = None,
):
    """Sample one uniform NNI neighbour (a symmetric proposal).

    If ``branch_lengths`` (length ``2n-2``, indexed by node id) is given, it is
    carried through the relabelling -- NNI reconnects subtrees without changing
    any edge length, so the returned lengths are the same values reindexed. This
    makes the joint ``(topology, branch)`` proposal symmetric with unit Hastings
    ratio.
    """
    edge_nodes = _internal_edge_nodes(taxon_count)
    c = edge_nodes[rng.integers(len(edge_nodes))]
    swap_b = bool(rng.integers(2))
    moved = _apply_nni(parent_indices, c, swap_b)
    new_parent, old_to_new = canonicalize_parent_indices(moved, taxon_count)
    if branch_lengths is None:
        return new_parent
    node_count = 2 * taxon_count - 1
    new_branch = np.empty_like(branch_lengths)
    for node in range(node_count - 1):  # non-root nodes carry a branch
        new_branch[old_to_new[node]] = branch_lengths[node]
    return new_parent, new_branch


# ---------------------------------------------------------------------------
# TransitionKernel over topologies
# ---------------------------------------------------------------------------
class TopologyMHResults(tp.NamedTuple):
    target_log_prob: tf.Tensor
    log_accept_ratio: tf.Tensor
    is_accepted: tf.Tensor
    proposed_state: tf.Tensor


class TopologyMetropolisHastings(tfp.mcmc.TransitionKernel):
    """Symmetric-proposal MH kernel over rooted topologies (eager).

    State is a rooted ``parent_indices`` int tensor of shape ``[2n-2]``. The
    proposal is a uniform rooted NNI, which is symmetric, so acceptance is
    ``min(1, exp(target(proposed) - target(current)))``.

    Parameters
    ----------
    target_log_prob_fn
        Callable mapping a ``parent_indices`` tensor to a scalar log density.
    taxon_count
        Number of taxa ``n``.
    seed
        Optional base seed for the internal NumPy RNG used to draw proposals.
    """

    def __init__(
        self,
        target_log_prob_fn: tp.Callable[[tf.Tensor], tf.Tensor],
        taxon_count: int,
        seed: tp.Optional[int] = None,
        name: str = "topology_mh",
    ):
        self._target_log_prob_fn = target_log_prob_fn
        self._taxon_count = int(taxon_count)
        self._rng = np.random.default_rng(seed)
        self._name = name

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def target_log_prob_fn(self):
        return self._target_log_prob_fn

    @property
    def taxon_count(self) -> int:
        return self._taxon_count

    def bootstrap_results(self, init_state) -> TopologyMHResults:
        init_state = tf.convert_to_tensor(init_state, dtype=tf.int32)
        tlp = tf.convert_to_tensor(self._target_log_prob_fn(init_state))
        return TopologyMHResults(
            target_log_prob=tlp,
            log_accept_ratio=tf.zeros([], dtype=tlp.dtype),
            is_accepted=tf.constant(True),
            proposed_state=init_state,
        )

    def one_step(
        self, current_state, previous_kernel_results, seed=None
    ) -> tp.Tuple[tf.Tensor, TopologyMHResults]:
        current_np = np.asarray(tf.convert_to_tensor(current_state).numpy())
        proposed_np = propose_nni(current_np, self._taxon_count, self._rng)
        proposed_state = tf.constant(proposed_np, dtype=tf.int32)

        current_tlp = previous_kernel_results.target_log_prob
        proposed_tlp = tf.convert_to_tensor(
            self._target_log_prob_fn(proposed_state)
        )
        log_accept_ratio = proposed_tlp - current_tlp  # symmetric proposal
        u = self._rng.random()
        is_accepted = bool(
            np.log(u) < float(log_accept_ratio.numpy())
        )
        if is_accepted:
            next_state = proposed_state
            next_tlp = proposed_tlp
        else:
            next_state = tf.convert_to_tensor(current_state, dtype=tf.int32)
            next_tlp = current_tlp
        results = TopologyMHResults(
            target_log_prob=next_tlp,
            log_accept_ratio=log_accept_ratio,
            is_accepted=tf.constant(is_accepted),
            proposed_state=proposed_state,
        )
        return next_state, results


def sample_topology_chain(
    kernel: TopologyMetropolisHastings,
    init_state: np.ndarray,
    num_results: int,
    num_burnin_steps: int = 0,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Drive a :class:`TopologyMetropolisHastings` kernel (eager loop).

    Mirrors the role of ``tfp.mcmc.sample_chain`` for the HMC code, but stays in
    eager Python because the NNI proposal is a discrete graph operation.

    Returns
    -------
    samples
        ``int32`` array ``[num_results, 2n-2]`` of post-burn-in topologies.
    is_accepted
        ``bool`` array ``[num_results]`` of acceptance flags.
    """
    state = tf.constant(np.asarray(init_state), dtype=tf.int32)
    results = kernel.bootstrap_results(state)
    for _ in range(num_burnin_steps):
        state, results = kernel.one_step(state, results)
    samples = np.empty((num_results, 2 * kernel.taxon_count - 2), dtype=np.int32)
    accepted = np.empty(num_results, dtype=bool)
    for i in range(num_results):
        state, results = kernel.one_step(state, results)
        samples[i] = state.numpy()
        accepted[i] = bool(results.is_accepted.numpy())
    return samples, accepted


# ---------------------------------------------------------------------------
# Joint phylogenetic MCMC over (topology, branch lengths)
# ---------------------------------------------------------------------------
class PhylogeneticMCMCResults(tp.NamedTuple):
    topologies: np.ndarray  # [num_results, 2n-2]
    branch_lengths: np.ndarray  # [num_results, 2n-2]
    log_posterior: np.ndarray  # [num_results]
    topology_accept_rate: float
    branch_accept_rate: float


def sample_phylogenetic_topologies(
    leaf_partials: np.ndarray,
    taxon_count: int,
    num_results: int,
    num_burnin_steps: int = 0,
    branch_prior_rate: float = 10.0,
    branch_proposal_scale: float = 0.2,
    init_parent_indices: tp.Optional[np.ndarray] = None,
    init_branch_lengths: tp.Optional[np.ndarray] = None,
    thin: int = 1,
    seed: tp.Optional[int] = None,
) -> PhylogeneticMCMCResults:
    """Metropolis-within-Gibbs over the joint JC posterior on rooted trees.

    The posterior is ``p(T, b | D) ∝ p(D | T, b) · Exp(b; rate)``. Each sweep does
    an NNI topology move (branch lengths carried through, so the Hastings ratio is
    unity and acceptance is the likelihood ratio) followed by a multiplicative
    random-walk update of the branch lengths.

    Parameters
    ----------
    leaf_partials
        ``[n, n_sites, 4]`` leaf state partials, ordered by taxon id.
    branch_prior_rate
        Rate of the i.i.d. exponential branch-length prior.
    branch_proposal_scale
        Std of the log-scale random walk on branch lengths.
    thin
        Keep every ``thin``-th sweep.
    """
    from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
    from treeflow.tree.topology.tensorflow_tree_topology import (
        numpy_topology_to_tensor,
    )
    from treeflow.vbpi.likelihood import make_jc_log_likelihood_fn

    rng = np.random.default_rng(seed)
    n = int(taxon_count)
    node_count = 2 * n - 1
    # tf.function-compiled likelihood (native op when built): traced once, then
    # called with each proposed topology's index tensors -- the fast path.
    log_likelihood_fn = make_jc_log_likelihood_fn(leaf_partials)

    def log_likelihood(parent, branch):
        topology = numpy_topology_to_tensor(
            NumpyTreeTopology(parent_indices=np.asarray(parent))
        )
        return float(
            log_likelihood_fn(
                topology, tf.constant(branch, dtype=DEFAULT_FLOAT_DTYPE_TF)
            ).numpy()
        )

    def log_prior(branch):
        # i.i.d. Exponential(rate): sum(log rate - rate*b)
        return float(np.sum(np.log(branch_prior_rate) - branch_prior_rate * branch))

    # Initial state.
    if init_parent_indices is None:
        # A simple caterpillar: leaves 0..n-1, internal n..2n-2 in a ladder.
        parent = np.empty(node_count - 1, dtype=np.int32)
        parent[0] = n
        parent[1] = n
        for k in range(2, n):
            parent[k] = n + k - 1
        for k in range(n, node_count - 1):
            parent[k] = k + 1
        parent, _ = canonicalize_parent_indices(parent, n)
    else:
        parent = np.asarray(init_parent_indices, dtype=np.int32).copy()
    if init_branch_lengths is None:
        branch = rng.exponential(1.0 / branch_prior_rate, size=node_count - 1).astype(
            DEFAULT_FLOAT_DTYPE_NP
        )
    else:
        branch = np.asarray(init_branch_lengths, dtype=DEFAULT_FLOAT_DTYPE_NP).copy()

    cur_ll = log_likelihood(parent, branch)
    cur_lp = cur_ll + log_prior(branch)

    topo_accepts = 0
    branch_accepts = 0
    total_sweeps = num_burnin_steps + num_results * thin

    kept_topologies = np.empty((num_results, node_count - 1), dtype=np.int32)
    kept_branches = np.empty((num_results, node_count - 1), dtype=DEFAULT_FLOAT_DTYPE_NP)
    kept_logpost = np.empty(num_results, dtype=np.float64)
    kept = 0

    for sweep in range(total_sweeps):
        # --- Topology NNI move (branch lengths carried; symmetric proposal). ---
        prop_parent, prop_branch = propose_nni(parent, n, rng, branch_lengths=branch)
        prop_ll = log_likelihood(prop_parent, prop_branch)
        if np.log(rng.random()) < (prop_ll - cur_ll):
            parent, branch, cur_ll = prop_parent, prop_branch, prop_ll
            cur_lp = cur_ll + log_prior(branch)
            topo_accepts += 1

        # --- Branch-length random-walk move (multiplicative / log-space). ---
        log_factor = rng.normal(0.0, branch_proposal_scale, size=branch.shape)
        prop_branch = branch * np.exp(log_factor)
        prop_ll = log_likelihood(parent, prop_branch)
        prop_lp = prop_ll + log_prior(prop_branch)
        # Jacobian of the log-space (multiplicative) proposal: sum(log_factor).
        log_hastings = float(np.sum(log_factor))
        if np.log(rng.random()) < (prop_lp - cur_lp + log_hastings):
            branch, cur_ll, cur_lp = prop_branch, prop_ll, prop_lp
            branch_accepts += 1

        if sweep >= num_burnin_steps and (sweep - num_burnin_steps) % thin == 0:
            kept_topologies[kept] = parent
            kept_branches[kept] = branch
            kept_logpost[kept] = cur_lp
            kept += 1

    return PhylogeneticMCMCResults(
        topologies=kept_topologies,
        branch_lengths=kept_branches,
        log_posterior=kept_logpost,
        topology_accept_rate=topo_accepts / max(total_sweeps, 1),
        branch_accept_rate=branch_accepts / max(total_sweeps, 1),
    )


__all__ = [
    "canonicalize_parent_indices",
    "rooted_nni_neighbours",
    "num_nni_neighbours",
    "propose_nni",
    "TopologyMetropolisHastings",
    "TopologyMHResults",
    "sample_topology_chain",
    "sample_phylogenetic_topologies",
    "PhylogeneticMCMCResults",
]
