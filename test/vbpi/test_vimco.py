import itertools

import numpy as np
import pytest
import tensorflow as tf

from treeflow.vbpi.vimco import reduce_logmeanexp, vimco_surrogate


def test_reduce_logmeanexp_value():
    x = tf.constant([0.0, np.log(2.0), np.log(3.0)])
    # mean(exp(x)) = (1 + 2 + 3)/3 = 2
    assert np.isclose(reduce_logmeanexp(x).numpy(), np.log(2.0))


def test_vimco_requires_two_samples():
    with pytest.raises(ValueError):
        vimco_surrogate(tf.zeros([3, 1]), tf.zeros([3, 1]))


def test_surrogate_value_equals_elbo():
    logw = tf.constant(np.random.default_rng(0).normal(size=(4, 5)))
    logq = tf.constant(np.random.default_rng(1).normal(size=(4, 5)))
    obj = vimco_surrogate(logw, logq)
    # surrogate value (not gradient) equals the reported bound
    assert np.allclose(obj.surrogate.numpy(), obj.elbo.numpy())
    assert np.allclose(
        obj.elbo.numpy(), reduce_logmeanexp(logw, axis=-1).numpy()
    )


@pytest.mark.parametrize("K", [2, 3, 4])
def test_vimco_gradient_matches_exact_expectation(K):
    """VIMCO's gradient is unbiased for the exact K-sample bound gradient.

    Ground truth: enumerate every length-K tuple of discrete samples, form the
    exact expectation E_q[logmeanexp_k log w_k] and differentiate it. This is the
    correct reference (a common-random-number finite difference is *not*, because
    categorical sampling is piecewise-constant in the logits).
    """
    M = 5
    rng = np.random.default_rng(0)
    log_p = tf.nn.log_softmax(tf.constant(rng.normal(size=M)))
    theta0 = tf.constant(rng.normal(size=M))

    combos = np.array(list(itertools.product(range(M), repeat=K)))
    theta_exact = tf.Variable(theta0)
    with tf.GradientTape() as tape:
        log_q = tf.nn.log_softmax(theta_exact)
        q = tf.exp(log_q)
        lw = tf.gather(log_p, combos) - tf.gather(log_q, combos)
        elbo_k = reduce_logmeanexp(lw, axis=-1)
        prob_combo = tf.reduce_prod(tf.gather(q, combos), axis=-1)
        exact = tf.reduce_sum(prob_combo * elbo_k)
    g_exact = tape.gradient(exact, theta_exact).numpy()

    n_rep = 400000
    zs = tf.random.stateless_categorical(
        tf.broadcast_to(theta0, [n_rep, M]), K, seed=[7, 2]
    )
    theta_mc = tf.Variable(theta0)
    with tf.GradientTape() as tape:
        log_q = tf.nn.log_softmax(theta_mc)
        lw = tf.gather(log_p, zs) - tf.gather(log_q, zs)
        obj = vimco_surrogate(lw, tf.gather(log_q, zs))
        surrogate = tf.reduce_mean(obj.surrogate)
    g_mc = tape.gradient(surrogate, theta_mc).numpy()

    assert np.max(np.abs(g_mc - g_exact)) < 0.01


def test_vimco_pathwise_only_matches_iwae_gradient():
    """With no discrete dependence, VIMCO reduces to the IWAE bound gradient."""
    rng = np.random.default_rng(3)
    a0 = tf.constant(rng.normal(size=(6,)))
    # log_weights depend on a reparameterised parameter only; log_q_discrete is
    # a constant (detached) so the score term vanishes.
    base = tf.constant(rng.normal(size=(6,)))
    a = tf.Variable(a0)
    with tf.GradientTape() as tape:
        logw = a[:, None] * base[None, :]  # [6, 6]
        obj = vimco_surrogate(logw, tf.stop_gradient(logw))
        loss = tf.reduce_sum(obj.surrogate)
    g_vimco = tape.gradient(loss, a).numpy()

    a2 = tf.Variable(a0)
    with tf.GradientTape() as tape:
        logw = a2[:, None] * base[None, :]
        iwae = tf.reduce_sum(reduce_logmeanexp(logw, axis=-1))
    g_iwae = tape.gradient(iwae, a2).numpy()
    assert np.allclose(g_vimco, g_iwae, atol=1e-9)
