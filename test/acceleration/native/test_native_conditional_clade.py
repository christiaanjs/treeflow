"""Tests for the native (C++) conditional clade topology ops.

Each native op is checked against the validated pure-Python / TensorFlow
references: the eager ``ConditionalCladeDistribution`` for sampling and
log-probability, and the NumPy topology operations for the index transforms.
The whole module is tagged ``native`` and auto-skips if the op cannot be built.
"""

import numpy as np
import pytest
import tensorflow as tf

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.acceleration.native import conditional_clade as native
from treeflow.conditional_clade.distribution import ConditionalCladeDistribution
from treeflow.conditional_clade.support import ConditionalCladeSupport
from treeflow.conditional_clade.tree_distribution import (
    ConditionalCladeTreeDistribution,
)
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology


def _support_constants(support):
    pow2n = 1 << support.taxon_count
    clade_offset = np.zeros(pow2n, np.int32)
    clade_count = np.zeros(pow2n, np.int32)
    for parent_idx, clade in enumerate(support.parent_clades):
        clade_offset[clade] = support.parent_offsets[parent_idx]
        clade_count[clade] = len(support.subsplits_by_parent[parent_idx])
    flat_child1 = np.array([s.child1 for s in support.flat_subsplits], np.int32)
    flat_child2 = np.array([s.child2 for s in support.flat_subsplits], np.int32)
    flat_parent = np.array(support.flat_parents, np.int32)
    return dict(
        clade_offset=clade_offset,
        clade_count=clade_count,
        flat_child1=flat_child1,
        flat_child2=flat_child2,
        flat_parent=flat_parent,
    )


def _make(n, seed=0):
    support = ConditionalCladeSupport(n)
    rng = np.random.default_rng(seed)
    logits = tf.constant(
        rng.standard_normal(support.subsplit_count), dtype=DEFAULT_FLOAT_DTYPE_TF
    )
    dist = ConditionalCladeDistribution(support, logits)
    return support, logits, dist, _support_constants(support)


# ----------------------------------------------------------------------
# Topology index transforms
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n", [3, 4, 5])
def test_native_child_and_preorder_match_numpy(n):
    support = ConditionalCladeSupport(n)
    parent_list = support.enumerate_parent_indices()
    batch = np.stack(parent_list).astype(np.int32)
    child = native.native_parent_indices_to_child_indices(batch, n).numpy()
    preorder = native.native_child_indices_to_preorder(child, n).numpy()
    for k, parent_indices in enumerate(parent_list):
        numpy_topology = NumpyTreeTopology(parent_indices=parent_indices)
        np.testing.assert_array_equal(child[k], numpy_topology.child_indices)
        np.testing.assert_array_equal(
            preorder[k], numpy_topology.preorder_indices
        )


# ----------------------------------------------------------------------
# Log-probability
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n", [3, 4, 5])
def test_native_log_prob_matches_eager(n):
    support, logits, dist, const = _make(n, seed=1)
    parent_list = support.enumerate_parent_indices()
    batch = np.stack(parent_list).astype(np.int32)
    cond = dist.conditional_log_probs()
    native_lp = native.native_topology_log_prob(
        cond, batch, const["flat_parent"], const["flat_child1"], n
    ).numpy()
    eager_lp = np.array([float(dist.log_prob(p)) for p in parent_list])
    np.testing.assert_allclose(native_lp, eager_lp, atol=1e-12)
    assert np.isclose(np.sum(np.exp(native_lp)), 1.0)


def test_native_log_prob_gradient_matches_autodiff():
    n = 5
    support, logits, _, const = _make(n, seed=2)
    parent_list = support.enumerate_parent_indices()
    batch = np.stack(parent_list).astype(np.int32)
    weights = tf.constant(
        np.random.default_rng(0).uniform(size=len(parent_list)),
        dtype=DEFAULT_FLOAT_DTYPE_TF,
    )

    var = tf.Variable(logits)
    dist = ConditionalCladeDistribution(support, var)

    with tf.GradientTape() as tape:
        cond = dist.conditional_log_probs()
        native_lp = native.native_topology_log_prob(
            cond, batch, const["flat_parent"], const["flat_child1"], n
        )
        loss = tf.reduce_sum(native_lp * weights)
    native_grad = tape.gradient(loss, var).numpy()

    with tf.GradientTape() as tape:
        ref_lp = dist.enumerate_log_probs()
        ref_loss = tf.reduce_sum(ref_lp * weights)
    ref_grad = tape.gradient(ref_loss, var).numpy()

    np.testing.assert_allclose(native_grad, ref_grad, atol=1e-10)


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------
def test_native_sampler_matches_exact_pmf():
    n = 4
    support, logits, dist, const = _make(n, seed=3)
    parent_list = support.enumerate_parent_indices()
    index_of = {tuple(p.tolist()): i for i, p in enumerate(parent_list)}
    exact = dist.enumerate_probs().numpy()

    n_samples = 40000
    seeds = tf.cast(
        tf.random.experimental.stateless_split(
            tf.constant([1, 2], tf.int32), n_samples
        ),
        tf.int32,
    )
    samples = native.native_sample_parent_indices(
        logits,
        seeds,
        const["clade_offset"],
        const["clade_count"],
        const["flat_child1"],
        const["flat_child2"],
        n,
    ).numpy()

    counts = np.zeros(len(parent_list))
    for row in samples:
        key = tuple(row.tolist())
        assert key in index_of  # every sample is a valid topology
        counts[index_of[key]] += 1
    empirical = counts / n_samples
    assert np.max(np.abs(empirical - exact)) < 0.02


def test_native_sampler_is_deterministic_given_seed():
    n = 5
    support, logits, dist, const = _make(n, seed=4)
    seeds = tf.constant([[7, 8], [7, 8], [9, 10]], tf.int32)
    args = (
        logits,
        seeds,
        const["clade_offset"],
        const["clade_count"],
        const["flat_child1"],
        const["flat_child2"],
        n,
    )
    a = native.native_sample_parent_indices(*args).numpy()
    b = native.native_sample_parent_indices(*args).numpy()
    np.testing.assert_array_equal(a, b)  # same seeds -> same topologies
    np.testing.assert_array_equal(a[0], a[1])  # identical seed rows match


# ----------------------------------------------------------------------
# Native vs TensorFlow path through the distribution
# ----------------------------------------------------------------------
def test_distribution_native_matches_tf_path():
    n = 5
    support = ConditionalCladeSupport(n)
    logits = tf.constant(
        np.random.default_rng(5).standard_normal(support.subsplit_count),
        dtype=DEFAULT_FLOAT_DTYPE_TF,
    )
    native_dist = ConditionalCladeTreeDistribution(support, logits, use_native=True)
    tf_dist = ConditionalCladeTreeDistribution(support, logits, use_native=False)
    assert native_dist._use_native and not tf_dist._use_native

    for parent_indices in support.enumerate_parent_indices():
        value = TensorflowTreeTopology(
            tf.constant(parent_indices, tf.int32), None, None
        )
        np.testing.assert_allclose(
            float(native_dist.log_prob(value)),
            float(tf_dist.log_prob(value)),
            atol=1e-12,
        )


def test_distribution_native_sample_in_tf_function():
    n = 5
    support = ConditionalCladeSupport(n)
    logits = tf.constant(
        np.random.default_rng(6).standard_normal(support.subsplit_count),
        dtype=DEFAULT_FLOAT_DTYPE_TF,
    )
    dist = ConditionalCladeTreeDistribution(support, logits, use_native=True)
    node_count = 2 * n - 1

    @tf.function
    def sample_and_log_prob(seed):
        topology = dist.sample(8, seed=seed)
        return topology, dist.log_prob(topology)

    topology, log_prob = sample_and_log_prob((13, 14))
    assert topology.parent_indices.shape == (8, node_count - 1)
    assert log_prob.shape == (8,)
    assert np.all(np.isfinite(log_prob.numpy()))
