import tensorflow as tf
from tensorflow_probability.python.distributions import Independent
from treeflow.distributions.markov_chain.postorder import PostorderNodeMarkovChain
from treeflow.distributions.markov_chain.linear_gaussian import (
    LinearGaussianPostorderNodeMarkovChain,
)
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from numpy.testing import assert_allclose


def test_LinearGaussianPostorderNodeMarkovChain_sample(
    hello_tensor_tree: TensorflowRootedTree,
):
    float_dtype = hello_tensor_tree.node_heights.dtype
    scale = tf.constant(0.1, dtype=float_dtype)
    node_means = tf.random.normal(
        tf.expand_dims(hello_tensor_tree.taxon_count - 1, 0), seed=1, dtype=float_dtype
    )
    dist = LinearGaussianPostorderNodeMarkovChain(
        hello_tensor_tree.topology, node_means, scale
    )
    sample_shape = (4,)
    res = dist.sample(sample_shape, seed=2)
    assert res.numpy().shape == sample_shape + (hello_tensor_tree.taxon_count - 1,)


def test_LinearGaussianPostorderNodeMarkovChain_log_prob(
    hello_tensor_tree: TensorflowRootedTree,
):
    float_dtype = hello_tensor_tree.node_heights.dtype
    scale = tf.constant(0.1, dtype=float_dtype)
    node_means = tf.random.normal(
        tf.expand_dims(hello_tensor_tree.taxon_count - 1, 0), seed=1, dtype=float_dtype
    )
    dist = LinearGaussianPostorderNodeMarkovChain(
        hello_tensor_tree.topology, node_means, scale
    )
    sample_shape = (4,)
    sample = dist.sample(sample_shape, seed=2)

    log_prob = dist.log_prob(sample)
    expected_log_prob = PostorderNodeMarkovChain.log_prob(dist, sample)
    assert_allclose(log_prob.numpy(), expected_log_prob.numpy())


def test_LinearGaussianPostorderNodeMarkovChain_sample_and_log_prob_batch(
    hello_tensor_tree: TensorflowRootedTree,
):
    float_dtype = hello_tensor_tree.node_heights.dtype
    scale = tf.constant(0.1, dtype=float_dtype)
    batch_shape = (4,)
    node_means = tf.random.normal(
        batch_shape + (hello_tensor_tree.taxon_count - 1,), seed=1, dtype=float_dtype
    )
    dist = Independent(
        LinearGaussianPostorderNodeMarkovChain(
            hello_tensor_tree.topology, node_means, scale
        ),
        reinterpreted_batch_ndims=len(batch_shape),
    )
    sample_shape = (5,)
    sample = dist.sample(sample_shape, seed=2)
    assert sample.numpy().shape == sample_shape + batch_shape + (
        hello_tensor_tree.taxon_count - 1,
    )

    log_prob = dist.log_prob(sample)
    assert log_prob.numpy().shape == sample_shape
