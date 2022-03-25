import pytest
import numpy as np
from tensorflow_probability.python.distributions import (
    TruncatedNormal,
    Uniform,
    Normal,
    Categorical,
    Mixture,
)
from treeflow.distributions.discretized import DiscretizedDistribution
from treeflow.distributions.discrete_parameter_mixture import DiscreteParameterMixture
from numpy.testing import assert_allclose


@pytest.fixture
def discrete_distribution(tensor_constant):
    return DiscretizedDistribution(
        4,
        TruncatedNormal(
            loc=tensor_constant(2.2),
            scale=(0.5),
            low=tensor_constant(1.0),
            high=tensor_constant(5.0),
        ),
    )


@pytest.fixture
def continuous_distribution_maker(tensor_constant):
    bandwidth = tensor_constant(1.0)
    return lambda x: Uniform(low=x - bandwidth, high=x + bandwidth)


def test_marginalized_discrete_sample(
    discrete_distribution, continuous_distribution_maker
):
    dist = DiscreteParameterMixture(
        discrete_distribution, continuous_distribution_maker
    )
    sample_shape = (17, 4)
    samples = dist.sample(sample_shape, seed=1).numpy()
    assert samples.shape == sample_shape
    assert np.all((samples > 0.0) & (samples < 6.0))


def test_marginalized_discrete_prob(discrete_distribution: DiscretizedDistribution):
    normal_distribution_maker = lambda x: Normal(loc=x, scale=1.0)
    values = discrete_distribution.support.numpy()
    components = [normal_distribution_maker(x) for x in values]
    cat = Categorical(discrete_distribution.probabilities)
    standard_mixture_dist = Mixture(cat, components)
    sample_shape = (17, 4)
    samples = standard_mixture_dist.sample(sample_shape, seed=1).numpy()
    dist = DiscreteParameterMixture(discrete_distribution, normal_distribution_maker)
    assert_allclose(
        dist.log_prob(samples).numpy(), standard_mixture_dist.log_prob(samples).numpy()
    )
