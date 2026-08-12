"""Tests for the treatment of simplex-constrained variables in the variational
approximation.

The default event space bijector for a Dirichlet distribution is
``SoftmaxCentered``, which appends a fixed zero coordinate before applying a
softmax. The final component of the constrained variable therefore carries no
unconstrained coordinate of its own. A factorised Gaussian in the unconstrained
space is not invariant to which component is placed last, and it underestimates
the dispersion of that pivot component. A full covariance Gaussian is invariant,
because changing the pivot is a linear reparameterisation of the unconstrained
coordinates.

These tests fit an approximation to a Dirichlet target, whose marginals are Beta
distributed with standard deviations known in closed form, and compare.
"""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd

from treeflow.model.approximation.mean_field import (
    get_mean_field_approximation,
    get_simplex_variable_names,
)

CONCENTRATION = np.array([0.39, 0.305, 0.081, 0.223]) * 400.0


def dirichlet_marginal_sd(concentration):
    """Exact standard deviation of each Dirichlet marginal."""
    total = concentration.sum()
    return np.sqrt(
        concentration * (total - concentration) / (total**2 * (total + 1.0))
    )


def build_model(concentration):
    return tfd.JointDistributionNamed(
        dict(frequencies=tfd.Dirichlet(tf.constant(concentration, tf.float64)))
    )


def fit(model, full_covariance_names, steps=1500, seed=1):
    tf.random.set_seed(seed)
    approximation, _ = get_mean_field_approximation(
        model,
        dtype=tf.float64,
        full_covariance_names=full_covariance_names,
    )
    optimizer = tf.optimizers.Adam(learning_rate=0.05)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            sample = approximation.sample(64)
            loss = tf.reduce_mean(
                approximation.log_prob(sample) - model.log_prob(sample)
            )
        variables = approximation.trainable_variables
        optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
        return loss

    for _ in range(steps):
        step()
    return approximation


def approximate_marginal_sd(approximation, n=40000, seed=2):
    tf.random.set_seed(seed)
    return (
        tf.math.reduce_std(approximation.sample(n)["frequencies"], axis=0)
        .numpy()
        .astype(np.float64)
    )


def test_simplex_variable_names_detects_dirichlet():
    model = build_model(CONCENTRATION)
    assert get_simplex_variable_names(model) == {"frequencies"}


def test_simplex_variable_names_ignores_unconstrained():
    model = tfd.JointDistributionNamed(
        dict(x=tfd.Normal(tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64)))
    )
    assert get_simplex_variable_names(model) == set()


def test_mean_field_underestimates_pivot_dispersion():
    """Documents the artifact that motivates the full covariance default."""
    model = build_model(CONCENTRATION)
    approximation = fit(model, full_covariance_names=None)
    ratio = approximate_marginal_sd(approximation) / dirichlet_marginal_sd(CONCENTRATION)
    # The pivot (last) component is markedly under-dispersed under mean field.
    assert ratio[-1] < 0.8
    # The remaining components are approximated far better.
    assert np.all(ratio[:-1] > 0.9)


def test_full_covariance_recovers_all_marginal_sds():
    model = build_model(CONCENTRATION)
    approximation = fit(model, full_covariance_names="simplex")
    ratio = approximate_marginal_sd(approximation) / dirichlet_marginal_sd(CONCENTRATION)
    np.testing.assert_allclose(ratio, np.ones_like(ratio), rtol=0.15)


@pytest.mark.parametrize("permutation", [[0, 1, 2, 3], [3, 1, 2, 0], [1, 3, 0, 2]])
def test_full_covariance_is_invariant_to_component_order(permutation):
    """The estimated dispersion of a component must not depend on its position."""
    permutation = np.asarray(permutation)
    model = build_model(CONCENTRATION[permutation])
    approximation = fit(model, full_covariance_names="simplex")
    ratio = approximate_marginal_sd(approximation) / dirichlet_marginal_sd(
        CONCENTRATION[permutation]
    )
    np.testing.assert_allclose(ratio, np.ones_like(ratio), rtol=0.15)
