"""Tests for the time-tree VBPI/MCMC components.

Node heights are represented through TreeFlow's ``NodeHeightRatioChainBijector``
and scored under the existing coalescent / Yule tree priors.
"""
from math import factorial

import numpy as np
import pytest
import tensorflow as tf

from treeflow.vbpi.clade import child_indices_from_parent_indices
from treeflow.vbpi.mcmc import (
    canonicalize_parent_indices,
    propose_nni_internal,
    sample_phylogenetic_time_trees,
)
from treeflow.vbpi.sbn import SubsplitBayesianNetwork
from treeflow.vbpi.support import SubsplitSupport, all_rooted_parent_indices
from treeflow.vbpi.timetree import (
    NodeHeightRatioModel,
    as_topology,
    build_time_tree,
    coalescent_prior,
    time_tree_jc_log_likelihood,
    yule_prior,
)

from vbpi_test_helpers import random_parent_indices, simulate_jc_alignment


# ---------------------------------------------------------------------------
# Tree building and priors
# ---------------------------------------------------------------------------
def test_build_time_tree_positive_branches():
    n = 5
    parent = random_parent_indices(n, np.random.default_rng(0))
    node_heights = tf.constant([0.1, 0.2, 0.35, 0.6], dtype=tf.float64)
    tree = build_time_tree(as_topology(parent), node_heights)
    assert np.all(tree.branch_lengths.numpy() > 0)


def test_priors_callable_and_differentiable():
    n = 5
    parent = random_parent_indices(n, np.random.default_rng(1))
    topology = as_topology(parent)
    heights = tf.Variable(tf.constant([0.1, 0.2, 0.35, 0.6], dtype=tf.float64))
    for prior in (
        coalescent_prior(n, tf.constant(1.0, tf.float64)),
        yule_prior(n, tf.constant(2.0, tf.float64)),
    ):
        with tf.GradientTape() as tape:
            lp = prior.log_prob(build_time_tree(topology, heights))
        grad = tape.gradient(lp, heights)
        assert np.isfinite(float(lp))
        assert grad is not None and np.all(np.isfinite(grad.numpy()))


def test_time_tree_likelihood_differentiable_in_heights():
    n = 5
    rng = np.random.default_rng(2)
    parent = random_parent_indices(n, rng)
    topology = as_topology(parent)
    leaf = np.eye(4)[rng.integers(0, 4, size=(n, 20))]
    heights = tf.Variable(tf.constant([0.1, 0.2, 0.35, 0.6], dtype=tf.float64))
    with tf.GradientTape() as tape:
        ll = time_tree_jc_log_likelihood(topology, tf.constant(leaf), heights)
    grad = tape.gradient(ll, heights)
    assert np.isfinite(float(ll))
    assert grad is not None and np.all(np.isfinite(grad.numpy()))


# ---------------------------------------------------------------------------
# VBPI node-height variational model
# ---------------------------------------------------------------------------
def test_height_model_shapes_and_valid_trees():
    n = 6
    support = SubsplitSupport.from_topologies(all_rooted_parent_indices(n), n)
    sbn = SubsplitBayesianNetwork(support)
    samples = sbn.sample_topologies(5, seed=0)
    topologies = [as_topology(samples.parent_indices[k]) for k in range(5)]
    model = NodeHeightRatioModel(support)
    heights, log_prob = model.sample_and_log_prob(
        tf.constant(samples.node_clade_ids), topologies, seed=(1, 2)
    )
    assert heights.shape == (5, n - 1)
    assert log_prob.shape == (5,)
    assert np.all(heights.numpy() > 0)
    # every sampled height vector is a valid (positive-branch) time tree
    for k in range(5):
        tree = build_time_tree(topologies[k], heights[k])
        assert np.all(tree.branch_lengths.numpy() > 0)


def test_height_model_reparameterised_gradient():
    n = 5
    support = SubsplitSupport.from_topologies(all_rooted_parent_indices(n), n)
    sbn = SubsplitBayesianNetwork(support)
    samples = sbn.sample_topologies(4, seed=3)
    topologies = [as_topology(samples.parent_indices[k]) for k in range(4)]
    model = NodeHeightRatioModel(support)
    with tf.GradientTape() as tape:
        heights, log_prob = model.sample_and_log_prob(
            tf.constant(samples.node_clade_ids), topologies, seed=(4, 5)
        )
        objective = tf.reduce_sum(heights) + tf.reduce_sum(log_prob)
    grads = tape.gradient(objective, [model.loc, model.scale_param])
    assert all(g is not None for g in grads)
    assert all(
        np.all(np.isfinite(tf.convert_to_tensor(g).numpy())) for g in grads
    )


# ---------------------------------------------------------------------------
# Time-tree NNI move carrying the height latent
# ---------------------------------------------------------------------------
def test_propose_nni_internal_permutes_latent():
    n = 5
    rng = np.random.default_rng(0)
    parent = random_parent_indices(n, rng)
    z = rng.normal(size=n - 1)
    new_parent, new_z = propose_nni_internal(parent, z, n, rng)
    # The latent values are carried (a permutation of the originals).
    assert sorted(new_z.tolist()) == pytest.approx(sorted(z.tolist()))
    # and the result is a valid topology
    for node, p in enumerate(new_parent):
        assert p > node


# ---------------------------------------------------------------------------
# Joint time-tree MCMC
# ---------------------------------------------------------------------------
def _num_rankings(parent, n):
    """Number of rankings (linear extensions) of a rooted tree's internal poset."""
    node_count = 2 * n - 1
    children = child_indices_from_parent_indices(parent, node_count)
    counts = {}

    def rec(node):
        if node < n:
            return 0
        total = 1
        for c in children[node]:
            total += rec(c)
        counts[node] = total
        return total

    rec(node_count - 1)
    prod = 1
    for v in counts.values():
        prod *= v
    return factorial(n - 1) // prod


def test_time_tree_mcmc_matches_coalescent_prior_marginal():
    # Constant (all-gap) likelihood: the posterior over topologies is the
    # coalescent prior marginal, which is proportional to the number of rankings
    # (linear extensions) of each tree -- NOT uniform.
    n = 4
    leaf = np.ones((n, 4, 4))
    res = sample_phylogenetic_time_trees(
        leaf, n, num_results=20000, num_burnin_steps=3000,
        prior="coalescent", pop_size=1.0, height_proposal_scale=0.4, seed=0,
    )
    topos = all_rooted_parent_indices(n)
    key = lambda p: tuple(canonicalize_parent_indices(p, n)[0].tolist())
    index = {key(t): i for i, t in enumerate(topos)}
    ranks = np.array([_num_rankings(t, n) for t in topos], dtype=float)
    expected = ranks / ranks.sum()
    counts = np.zeros(len(topos))
    for s in res.topologies:
        counts[index[key(s)]] += 1
    freq = counts / counts.sum()
    assert np.max(np.abs(freq - expected)) < 0.03


def _unrooted_signature(parent, n):
    from treeflow.vbpi.clade import node_clades

    full = frozenset(range(n))
    return frozenset(
        frozenset({c, full - c})
        for c in node_clades(parent, n)
        if 2 <= len(c) <= n - 2
    )


def test_time_tree_mcmc_recovers_true_topology():
    n = 5
    rng = np.random.default_rng(7)
    true_tree = random_parent_indices(n, rng)
    node_count = 2 * n - 1
    # Ultrametric heights: leaves at 0, internal nodes at increasing times (short
    # branches => strong phylogenetic signal). node ids are children<parent, so
    # sorted internal heights respect the ordering.
    heights = np.concatenate(
        [np.zeros(n), np.sort(rng.uniform(0.02, 0.12, n - 1))]
    )
    branch = np.array(
        [heights[true_tree[i]] - heights[i] for i in range(node_count - 1)]
    )
    leaf = simulate_jc_alignment(true_tree, branch, 400, n, rng)

    res = sample_phylogenetic_time_trees(
        leaf, n, num_results=12000, num_burnin_steps=3000,
        prior="coalescent", pop_size=0.2, height_proposal_scale=0.4, seed=1,
    )
    from collections import Counter

    counts = Counter(_unrooted_signature(s, n) for s in res.topologies)
    mode_sig, mode_count = counts.most_common(1)[0]
    assert mode_sig == _unrooted_signature(true_tree, n)
    assert mode_count / len(res.topologies) > 0.5
