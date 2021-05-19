import tensorflow as tf
import tensorflow_probability as tfp
import pytest
from treeflow.clock_approx import (
    get_lognormal_loc_conjugate_posterior,
    get_lognormal_precision_conjugate_posterior,
)
import treeflow


@pytest.fixture
def loc_prior():
    return tfp.distributions.Normal(loc=-8.12, scale=0.615)


@pytest.fixture
def precision_prior():
    return tfp.distributions.Gamma(concentration=2.03, rate=0.0559)


# loc =-7.82, -8.02
# precision = 50.0, 48.0


@pytest.fixture
def rates():
    return tf.convert_to_tensor(
        [0.00031997, 0.00038053, 0.00039979], dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF
    )


def test_lognormal_loc_conjugate_posterior(loc_prior, rates):
    posterior = get_lognormal_loc_conjugate_posterior(loc_prior)(rates)
    unnormalised_posterior = lambda loc: 