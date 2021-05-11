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


def test_get_params_for_quantiles_lognormal_conjugate():
    cov_quantiles = np.array([0.1, 0.5])
    mean_quantiles = np.array([1e-4, 1e-3])
    result, opt_res = get_params_for_quantiles_lognormal_conjugate(
        cov_quantiles, mean_quantiles, probs=PROBS
    )
    assert opt_res["loc"].success
    assert opt_res["precision"].success

    def lognormal_cov(precision):
        variance = 1.0 / precision
        return np.sqrt(np.exp(variance) - 1)

    def lognormal_mean(
        loc, precision
    ):  # Increasing function of mean, decreasing function of precision
        variance = 1.0 / precision
        return np.exp(loc + variance / 2.0)

    precision_quantiles_res = (
        tfp.distributions.Gamma(**result["precision"]).quantile(PROBS).numpy()
    )
    cov_quantiles_res = lognormal_cov(precision_quantiles_res[::-1])
    assert_allclose(cov_quantiles_res, cov_quantiles)

    loc_quantiles_res = (
        tfp.distributions.Normal(**result["loc"]).quantile(PROBS).numpy()
    )
    mean_quantiles_res = lognormal_mean(
        loc_quantiles_res, precision_quantiles_res[::-1]
    )
    assert_allclose(
        mean_quantiles_res, mean_quantiles, atol=1e-3
    )  # Tricky to get close with very small values
