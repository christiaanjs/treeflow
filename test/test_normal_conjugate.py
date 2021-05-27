import tensorflow as tf
import tensorflow_probability as tfp
import pytest
from treeflow.priors import get_normal_conjugate_prior_dict, precision_to_scale
from treeflow.clock_approx import get_normal_conjugate_posterior_dict
import treeflow


@pytest.fixture
def normal_gamma_params():
    return tfp.distributions.Normal(
        loc=-8.12, precision_scale=0.615, concentration=2.03, rate=0.0559
    )


# loc =-7.82, -8.02
# precision = 50.0, 48.0


@pytest.fixture
def normal_observations():
    return tf.convert_to_tensor(
        [0.00031997, 0.00038053, 0.00039979], dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF
    )


def test_normal_conjugate_posterior(normal_gamma_params, normal_observations):
    sample_shape = tf.shape(normal_observations)
    model = tfp.distributions.JointDistributionNamed(
        dict(
            get_normal_conjugate_prior_dict(**normal_gamma_params),
            x=lambda loc, precision: tfp.distributions.Sample(
                tfp.distributions.Normal(loc=loc, scale=precision_to_scale(precision)),
                sample_shape=sample_shape,
            ),
        )
    )
    posterior_dict = get_normal_conjugate_posterior_dict(**normal_gamma_params)
    posterior = tfp.distributions.JointDistributionNamed(
        dict(
            precision=posterior_dict["precision"](normal_observations),
            loc=lambda precision: posterior_dict["loc"](normal_observations, precision),
        )
    )
    assert False  # TODO: Marginal likelihood test?
