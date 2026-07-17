import numpy as np
import pytest
import tensorflow as tf

from treeflow.vbpi.clade import node_clades
from treeflow.vbpi.mcmc import (
    canonicalize_parent_indices,
    num_nni_neighbours,
    propose_nni,
    rooted_nni_neighbours,
    sample_phylogenetic_topologies,
    sample_topology_chain,
    TopologyMetropolisHastings,
)
from treeflow.vbpi.sbn import SubsplitBayesianNetwork
from treeflow.vbpi.support import SubsplitSupport, all_rooted_parent_indices

from vbpi_test_helpers import random_parent_indices, simulate_jc_alignment


# ---------------------------------------------------------------------------
# NNI moves and canonicalisation
# ---------------------------------------------------------------------------
def _canon_key(parent, n):
    return tuple(canonicalize_parent_indices(parent, n)[0].tolist())


@pytest.mark.parametrize("n", [4, 5, 6])
def test_nni_neighbour_count(n):
    rng = np.random.default_rng(n)
    t = random_parent_indices(n, rng)
    assert len(rooted_nni_neighbours(t, n)) == num_nni_neighbours(n) == 2 * (n - 2)


@pytest.mark.parametrize("n", [4, 5, 6])
def test_nni_neighbours_are_valid_and_symmetric(n):
    rng = np.random.default_rng(n + 1)
    valid = {_canon_key(t, n) for t in all_rooted_parent_indices(n)}
    for _ in range(20):
        t = random_parent_indices(n, rng)
        tk = _canon_key(t, n)
        for nb in rooted_nni_neighbours(t, n):
            assert _canon_key(nb, n) in valid
            # symmetric: t is a neighbour of nb
            back = {_canon_key(x, n) for x in rooted_nni_neighbours(nb, n)}
            assert tk in back


def test_canonical_form_is_unique_per_tree():
    n = 5
    topos = all_rooted_parent_indices(n)
    keys = {_canon_key(t, n) for t in topos}
    assert len(keys) == len(topos)


def test_canonicalize_preserves_clades():
    n = 6
    rng = np.random.default_rng(3)
    t = random_parent_indices(n, rng)
    canon, _ = canonicalize_parent_indices(t, n)
    orig_clades = set(node_clades(t, n))
    canon_clades = set(node_clades(canon, n))
    assert orig_clades == canon_clades


def test_propose_nni_carries_branch_lengths():
    n = 5
    rng = np.random.default_rng(0)
    t = random_parent_indices(n, rng)
    branch = rng.uniform(0.1, 0.5, size=2 * n - 2)
    new_t, new_branch = propose_nni(t, n, rng, branch_lengths=branch)
    # NNI reconnects subtrees without changing any edge length, so the multiset
    # of branch lengths is preserved.
    assert np.allclose(sorted(branch), sorted(new_branch))


# ---------------------------------------------------------------------------
# TransitionKernel over topologies
# ---------------------------------------------------------------------------
def test_kernel_targets_enumerable_distribution():
    n = 5
    topos = all_rooted_parent_indices(n)
    support = SubsplitSupport.from_topologies(topos, n)
    sbn = SubsplitBayesianNetwork(support)
    sbn.logits.assign(
        tf.constant(np.random.default_rng(2).normal(size=support.num_candidates))
    )
    log_probs = sbn.log_prob_of_topologies(topos).numpy()
    probs = np.exp(log_probs)
    cand = support.batch_topology_candidate_indices(topos)
    key_to_index = {tuple(sorted(c)): i for i, c in enumerate(cand)}

    def target(parent):
        idx = key_to_index[
            tuple(sorted(support.topology_candidate_indices(np.asarray(parent))))
        ]
        return tf.constant(float(log_probs[idx]))

    kernel = TopologyMetropolisHastings(target, n, seed=1)
    samples, accepted = sample_topology_chain(
        kernel, topos[0], num_results=40000, num_burnin_steps=2000
    )
    counts = np.zeros(len(topos))
    for s in samples:
        counts[
            key_to_index[tuple(sorted(support.topology_candidate_indices(s)))]
        ] += 1
    freq = counts / counts.sum()
    assert np.max(np.abs(freq - probs)) < 0.02
    assert 0.0 < accepted.mean() < 1.0


def test_kernel_is_calibrated():
    kernel = TopologyMetropolisHastings(lambda p: tf.constant(0.0), 5)
    assert kernel.is_calibrated is True


# ---------------------------------------------------------------------------
# Joint phylogenetic MCMC over (topology, branch lengths)
# ---------------------------------------------------------------------------
def test_joint_sampler_recovers_uniform_under_constant_likelihood():
    n = 4
    leaf = np.ones((n, 4, 4))  # all-gap partials => constant likelihood
    res = sample_phylogenetic_topologies(
        leaf, n, num_results=15000, num_burnin_steps=2000,
        branch_prior_rate=10.0, branch_proposal_scale=0.3, seed=0,
    )
    topos = all_rooted_parent_indices(n)
    index = {_canon_key(t, n): i for i, t in enumerate(topos)}
    counts = np.zeros(len(topos))
    for s in res.topologies:
        counts[index[_canon_key(s, n)]] += 1
    freq = counts / counts.sum()
    assert np.max(np.abs(freq - 1.0 / len(topos))) < 0.025
    # branch marginal recovers the Exp(rate) prior mean 1/rate = 0.1
    assert abs(res.branch_lengths.mean() - 0.1) < 0.02


def _unrooted_signature(parent, n):
    """Set of non-trivial splits of the tree (its unrooted topology id)."""
    clades = node_clades(parent, n)
    full = frozenset(range(n))
    splits = set()
    for clade in clades:
        if 2 <= len(clade) <= n - 2:
            splits.add(frozenset({clade, full - clade}))
    return frozenset(splits)


def test_joint_sampler_recovers_true_topology():
    n = 5
    rng = np.random.default_rng(7)
    true_tree = random_parent_indices(n, rng)
    branch = rng.uniform(0.05, 0.2, size=2 * n - 2)  # short => strong signal
    leaf = simulate_jc_alignment(true_tree, branch, 400, n, rng)

    res = sample_phylogenetic_topologies(
        leaf, n, num_results=15000, num_burnin_steps=3000,
        branch_prior_rate=10.0, branch_proposal_scale=0.3, seed=1,
    )
    true_sig = _unrooted_signature(true_tree, n)
    from collections import Counter

    counts = Counter(_unrooted_signature(s, n) for s in res.topologies)
    mode_sig, mode_count = counts.most_common(1)[0]
    assert mode_sig == true_sig
    assert mode_count / len(res.topologies) > 0.5
