"""Tests for the graph-compatible tfp distribution over TensorflowTreeTopology."""

import numpy as np
import pytest
import tensorflow as tf

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.conditional_clade.support import ConditionalCladeSupport
from treeflow.conditional_clade.distribution import ConditionalCladeDistribution
from treeflow.conditional_clade.tree_distribution import (
    ConditionalCladeTreeDistribution,
)
from treeflow.conditional_clade import tensor_ops
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology


def random_logits(support, seed=0):
    rng = np.random.default_rng(seed)
    return tf.constant(
        rng.standard_normal(support.subsplit_count), dtype=DEFAULT_FLOAT_DTYPE_TF
    )


def make_dist(n, seed=0):
    support = ConditionalCladeSupport(n)
    return ConditionalCladeTreeDistribution(support, random_logits(support, seed))


def topology_value(parent_indices):
    return TensorflowTreeTopology(
        parent_indices=tf.constant(parent_indices, tf.int32),
        child_indices=None,
        preorder_indices=None,
    )


# ----------------------------------------------------------------------
# Pure tensor ops vs the validated NumPy topology implementation
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n", [3, 4, 5])
def test_tensor_child_and_preorder_match_numpy(n):
    support = ConditionalCladeSupport(n)
    node_count = 2 * n - 1
    for parent_indices in support.enumerate_parent_indices():
        child = tensor_ops.parent_indices_to_child_indices(
            tf.constant(parent_indices, tf.int32), node_count
        ).numpy()
        preorder = tensor_ops.child_indices_to_preorder(
            tf.constant(child, tf.int32), node_count
        ).numpy()
        numpy_topology = NumpyTreeTopology(parent_indices=parent_indices)
        np.testing.assert_array_equal(child, numpy_topology.child_indices)
        np.testing.assert_array_equal(preorder, numpy_topology.preorder_indices)


# ----------------------------------------------------------------------
# Sample shapes / structure
# ----------------------------------------------------------------------
def test_sample_returns_topology_with_correct_shapes():
    n = 5
    dist = make_dist(n)
    node_count = 2 * n - 1
    sample = dist.sample(7, seed=(1, 2))
    assert isinstance(sample, TensorflowTreeTopology)
    assert sample.parent_indices.shape == (7, node_count - 1)
    assert sample.child_indices.shape == (7, node_count, 2)
    assert sample.preorder_indices.shape == (7, node_count)


def test_single_sample_shape():
    n = 4
    dist = make_dist(n)
    node_count = 2 * n - 1
    sample = dist.sample(seed=(3, 4))
    assert sample.parent_indices.shape == (node_count - 1,)


def test_sampled_topologies_are_valid():
    n = 5
    dist = make_dist(n)
    support = dist.support
    valid = {tuple(pi.tolist()) for pi in support.enumerate_parent_indices()}
    sample = dist.sample(50, seed=(7, 8))
    for pi in sample.parent_indices.numpy():
        assert tuple(pi.tolist()) in valid


# ----------------------------------------------------------------------
# log_prob correctness
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n", [3, 4, 5])
def test_log_prob_matches_eager_reference(n):
    support = ConditionalCladeSupport(n)
    logits = random_logits(support, seed=1)
    dist = ConditionalCladeTreeDistribution(support, logits)
    reference = ConditionalCladeDistribution(support, logits)
    for parent_indices in support.enumerate_parent_indices():
        graph_lp = float(dist.log_prob(topology_value(parent_indices)))
        eager_lp = float(reference.log_prob(parent_indices))
        assert np.isclose(graph_lp, eager_lp)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_log_prob_normalised(n):
    dist = make_dist(n, seed=2)
    total = sum(
        float(tf.exp(dist.log_prob(topology_value(pi))))
        for pi in dist.support.enumerate_parent_indices()
    )
    assert np.isclose(total, 1.0)


def test_batched_log_prob():
    dist = make_dist(5, seed=3)
    sample = dist.sample(6, seed=(9, 9))
    batched = dist.log_prob(sample).numpy()
    per_element = np.array(
        [
            float(dist.ccd.log_prob(sample.parent_indices[i].numpy()))
            for i in range(6)
        ]
    )
    np.testing.assert_allclose(batched, per_element)


# ----------------------------------------------------------------------
# Graph mode (tf.function)
# ----------------------------------------------------------------------
def test_sample_and_log_prob_in_tf_function():
    n = 5
    dist = make_dist(n, seed=4)
    node_count = 2 * n - 1

    @tf.function
    def sample_and_log_prob(seed):
        topology = dist.sample(4, seed=seed)
        return topology, dist.log_prob(topology)

    topology, log_prob = sample_and_log_prob((11, 12))
    assert topology.parent_indices.shape == (4, node_count - 1)
    assert log_prob.shape == (4,)
    assert np.all(np.isfinite(log_prob.numpy()))


def test_graph_sample_matches_exact_pmf():
    n = 4
    dist = make_dist(n, seed=5)
    parent_list = dist.support.enumerate_parent_indices()
    index_of = {tuple(p.tolist()): i for i, p in enumerate(parent_list)}
    exact = dist.ccd.enumerate_probs().numpy()

    @tf.function
    def draw(seed):
        return dist.sample(seed=seed).parent_indices

    n_samples = 20000
    seeds = tf.random.experimental.stateless_split(
        tf.constant([5, 6], tf.int32), n_samples
    )
    counts = np.zeros(len(parent_list))
    for k in range(n_samples):
        pi = draw(seeds[k]).numpy()
        counts[index_of[tuple(pi.tolist())]] += 1
    empirical = counts / n_samples
    assert np.max(np.abs(empirical - exact)) < 0.02


# ----------------------------------------------------------------------
# Differentiability
# ----------------------------------------------------------------------
def test_log_prob_differentiable_in_graph_mode():
    n = 5
    support = ConditionalCladeSupport(n)
    logits = tf.Variable(random_logits(support, seed=6))
    dist = ConditionalCladeTreeDistribution(support, logits)
    parent_indices = support.enumerate_parent_indices()[0]

    @tf.function
    def log_prob_of_fixed():
        return dist.log_prob(topology_value(parent_indices))

    with tf.GradientTape() as tape:
        lp = log_prob_of_fixed()
    grad = tape.gradient(lp, logits)
    assert grad is not None
    assert np.any(grad.numpy() != 0)


def test_score_function_gradient_in_graph_mode():
    """REINFORCE through the graph-mode distribution reduces KL to a target."""
    n = 4
    support = ConditionalCladeSupport(n)
    target = ConditionalCladeDistribution(support, random_logits(support, seed=20))
    log_p_cond = target.conditional_log_probs()

    logits = tf.Variable(random_logits(support, seed=21))
    dist = ConditionalCladeTreeDistribution(support, logits)
    optimizer = tf.optimizers.Adam(0.2)

    def kl():
        return float(dist.ccd.exact_kl_divergence(target).numpy())

    initial = kl()

    @tf.function
    def step(seed):
        topology = dist.sample(32, seed=seed)
        with tf.GradientTape() as tape:
            log_q = dist.log_prob(topology)
            # reward = log q - log p (reverse-KL integrand)
            log_p = tf.map_fn(
                lambda pi: tensor_ops.topology_log_prob(
                    log_p_cond, pi, n, 2 * n - 1, dist._flat_index_table, 1 << n
                ),
                topology.parent_indices,
                fn_output_signature=log_q.dtype,
            )
            cost = log_q - log_p
            baseline = tf.reduce_mean(tf.stop_gradient(cost))
            surrogate = tf.reduce_mean(
                tf.stop_gradient(cost - baseline) * log_q + cost
            )
        grad = tape.gradient(surrogate, logits)
        optimizer.apply_gradients([(grad, logits)])

    seeds = tf.random.experimental.stateless_split(
        tf.constant([1, 1], tf.int32), 150
    )
    for k in range(150):
        step(seeds[k])
    assert kl() < initial
