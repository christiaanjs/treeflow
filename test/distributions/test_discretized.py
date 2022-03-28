import tensorflow as tf
from tensorflow_probability.python.distributions import Normal, Gamma
from treeflow.distributions.discretized import DiscretizedDistribution
from numpy.testing import assert_allclose


def test_discretized_log_prob():
    base_dist = Normal(2.0, 2.0)
    k = 5
    discretized = DiscretizedDistribution(k, base_dist)
    quantiles = discretized.support
    other_length = 2
    first_others = quantiles[:other_length] - 1e-3
    last_others = quantiles[-other_length:] + 1e-3
    x = tf.concat([first_others, quantiles, last_others], axis=0)
    res = discretized.prob(x).numpy()
    assert res.shape == (2 * other_length + k,)
    mass = 1.0 / k
    assert_allclose(res[:other_length], 0.0)
    assert_allclose(res[-other_length:], 0.0)
    assert_allclose(res[other_length:-other_length], mass)


def test_discretized_sample():
    base_dist = Gamma(2.0, 2.0)
    k = 6
    discretized = DiscretizedDistribution(k, base_dist)
    mass = 1.0 / k
    sample_shape = (3, 2)
    sample = discretized.sample(sample_shape, seed=1)
    prob = discretized.prob(sample).numpy()
    assert prob.shape == sample_shape
    assert_allclose(prob, mass)


def test_discretized_sample_and_log_prob_batch(tensor_constant):
    batch_size = 3
    base_rate = tensor_constant(2.0)
    rate = base_rate + tf.range(batch_size, dtype=base_rate.dtype)
    base_dist = Gamma(tensor_constant(2.0), rate)
    k = 6
    discretized = DiscretizedDistribution(k, base_dist)
    mass = 1.0 / k
    sample_shape = 2
    sample = discretized.sample(sample_shape, seed=1)
    prob = discretized.prob(sample).numpy()
    assert prob.shape == (sample_shape, batch_size)
    assert_allclose(prob, mass)


def test_discretized_batch_shape(tensor_constant):
    batch_size = 3
    base_rate = tensor_constant(2.0)
    rate = base_rate + tf.range(batch_size, dtype=base_rate.dtype)
    base_dist = Gamma(tensor_constant(2.0), rate)
    discretized = DiscretizedDistribution(2, base_dist)
    batch_shape = discretized.batch_shape_tensor()
    assert tuple(batch_shape.numpy()) == (batch_size,)
