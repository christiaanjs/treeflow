import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import Normal
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from treeflow.distributions.markov_chain.postorder import PostorderNodeMarkovChain


def test_PostorderNodeMarkovChain_sample(hello_tensor_tree: TensorflowRootedTree):
    float_dtype = hello_tensor_tree.node_heights.dtype
    scale = tf.constant(0.1, dtype=float_dtype)
    node_means = tf.random.normal(
        tf.expand_dims(hello_tensor_tree.taxon_count - 1, 0), seed=1, dtype=float_dtype
    )
    dist = PostorderNodeMarkovChain(
        hello_tensor_tree.topology,
        lambda input, children: Normal(input + tf.reduce_sum(children, axis=0), scale),
        node_means,
        childless_init=tf.zeros((0,), float_dtype),
    )
    sample_shape = (4,)
    res = dist.sample(sample_shape, seed=2)
    assert res.numpy().shape == sample_shape + (hello_tensor_tree.taxon_count - 1,)


def test_PostorderNodeMarkovChain_log_prob(hello_tensor_tree: TensorflowRootedTree):
    float_dtype = hello_tensor_tree.node_heights.dtype
    scale = tf.constant(0.1, dtype=float_dtype)
    node_means = tf.random.normal(
        tf.expand_dims(hello_tensor_tree.taxon_count - 1, 0), seed=1, dtype=float_dtype
    )
    dist = PostorderNodeMarkovChain(
        hello_tensor_tree.topology,
        lambda input, children: Normal(input + tf.reduce_sum(children, axis=0), scale),
        node_means,
        childless_init=tf.zeros((0,), float_dtype),
    )
    sample_shape = (4,)
    samples = dist.sample(sample_shape, seed=2)
    res = dist.log_prob(samples)
    assert res.numpy().shape == sample_shape
    assert all(np.isfinite(res.numpy()))
