from unicodedata import category
import tensorflow as tf
from tensorflow_probability.python.distributions import Normal, Gamma
from treeflow.distributions.discretized import DiscretizedDistribution
from numpy.testing import assert_allclose


def test_discretized_log_prob():
    base_dist = Normal(2.0, 2.0)
    k = 5
    discretized = DiscretizedDistribution(k, base_dist)
    quantiles = discretized.quantiles
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
