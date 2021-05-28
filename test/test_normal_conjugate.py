from numpy.testing._private.utils import assert_allclose
import tensorflow as tf
import tensorflow_probability as tfp
import pytest
import treeflow
from treeflow.priors import (
    get_normal_conjugate_prior_dict,
    precision_to_scale,
    REAL_BOUNDS,
    NONNEGATIVE_BOUNDS,
)
from treeflow.clock_approx import get_normal_conjugate_posterior_dict
import treeflow
from scipy.integrate import dblquad
import numpy as np
from scipy.optimize import minimize


def t(x):
    return tf.convert_to_tensor(x, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF)


@pytest.fixture
def normal_gamma_params():
    # return dict(
    #     loc=t(-8.12), precision_scale=t(0.615), concentration=t(2.03), rate=t(0.0559)
    # )
    return dict(
        loc=t(-8.12), precision_scale=t(0.615), concentration=t(2.03), rate=t(0.0559)
    )


@pytest.fixture
def normal_gamma_samples():
    return dict(precision=t([33.59, 40.17]), loc=t([-7.930, -7.833]))


@pytest.fixture
def normal_observations():
    return t(
        [
            -7.970633,
            -8.029144,
            -7.6408343,
            -7.7481856,
            -7.6324663,
            -7.988668,
            -8.041067,
            -8.075005,
            -7.5542517,
            -7.9619846,
            -7.5677724,
            -8.211376,
            -8.142333,
            -8.156536,
            -7.7894654,
            -7.728593,
            -7.8994317,
            -8.117426,
            -7.9625087,
            -7.8603334,
        ]
    )


@pytest.fixture
def model(normal_observations, normal_gamma_params):
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
    return model


@pytest.fixture
def unnormalised_posterior_log_prob(model, normal_observations):
    return tf.function(lambda **kwargs: model.log_prob(x=normal_observations, **kwargs))


@pytest.fixture
def conjugate_posterior(normal_gamma_params, normal_observations):
    posterior_dict = get_normal_conjugate_posterior_dict(**normal_gamma_params)
    conjugate_posterior = tfp.distributions.JointDistributionNamed(
        dict(
            precision=posterior_dict["precision"](normal_observations),
            loc=lambda precision: posterior_dict["loc"](normal_observations, precision),
        )
    )
    return conjugate_posterior


@pytest.mark.skip(reason="Runtime of quadrature is too great")
def test_normal_conjugate_posterior_marginal_likelihood(
    normal_gamma_samples,
    unnormalised_posterior_log_prob,
    conjugate_posterior,
):
    def unnormalised_posterior_prob_flat(precision, loc):
        return np.exp(unnormalised_posterior_log_prob(precision=precision, loc=loc))

    marginal_likelihood, abserr, infodict, message = dblquad(
        unnormalised_posterior_prob_flat,
        -np.inf,
        np.inf,
        lambda x: 0.0,
        lambda x: np.inf,
    )

    assert_allclose(
        conjugate_posterior.log_prob(**normal_gamma_samples),
        unnormalised_posterior_log_prob(**normal_gamma_samples)
        - np.log(marginal_likelihood),
    )


def test_normal_conjugate_posterior_constant_normalisation_constant(
    unnormalised_posterior_log_prob, conjugate_posterior, normal_gamma_samples
):
    log_marginal_likelihoods = unnormalised_posterior_log_prob(
        **normal_gamma_samples
    ) - conjugate_posterior.log_prob(**normal_gamma_samples)
    assert_allclose(log_marginal_likelihoods[0], log_marginal_likelihoods[1])


def test_normal_conjugate_posterior_mode(
    unnormalised_posterior_log_prob, conjugate_posterior, normal_gamma_samples
):
    init = np.array(
        [normal_gamma_samples["precision"][0], normal_gamma_samples["loc"][0]]
    )

    def optimise_tf_log_prob(log_prob):
        @tf.function
        def neg_log_prob_and_grad_flat(x):
            val = -log_prob(precision=x[0], loc=x[1])
            return val, tf.gradients(val, x)

        def neg_log_prob_and_grad_flat_numpy(x):
            val, grad = neg_log_prob_and_grad_flat(x)
            return val.numpy(), grad.numpy()

        res = minimize(
            neg_log_prob_and_grad_flat,
            init,
            method="L-BFGS-B",
            jac=True,
            bounds=[NONNEGATIVE_BOUNDS, REAL_BOUNDS],
        )
        assert res.success
        return res.x

    expected_mode = optimise_tf_log_prob(unnormalised_posterior_log_prob)
    posterior_mode = optimise_tf_log_prob(conjugate_posterior.log_prob)
    assert_allclose(posterior_mode, expected_mode)
