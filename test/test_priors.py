import pytest
from numpy.testing import assert_allclose
from treeflow.priors import (
    get_params_for_quantiles,
    get_params_for_quantiles_lognormal_conjugate,
)
import tensorflow_probability as tfp
import numpy as np

PROBS = [0.025, 0.975]


@pytest.mark.parametrize(
    "dist_class,param_dict",
    [
        (tfp.distributions.Normal, dict(loc=-0.3, scale=1.2)),
        (tfp.distributions.Gamma, dict(concentration=2.03, rate=0.056)),
    ],
)
def test_get_params_for_quantiles(dist_class, param_dict):
    dist = dist_class(**param_dict)
    quantiles = dist.quantile(PROBS)
    param_result, opt_res = get_params_for_quantiles(dist_class, quantiles, probs=PROBS)
    assert opt_res.success
    dist_result = dist_class(**param_result)
    assert_allclose(quantiles, dist_result.quantile(PROBS))
