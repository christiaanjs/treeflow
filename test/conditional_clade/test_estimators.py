import numpy as np
import tensorflow as tf

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.conditional_clade.distribution import ConditionalCladeDistribution
from treeflow.conditional_clade.estimators import (
    gumbel_softmax,
    leave_one_out_baseline,
    sample_relaxed_cost,
    score_function_surrogate,
    straight_through_categorical,
    vimco_surrogate,
)
from treeflow.conditional_clade.support import ConditionalCladeSupport


def random_logits(support, seed=0):
    rng = np.random.default_rng(seed)
    return tf.constant(
        rng.standard_normal(support.subsplit_count), dtype=DEFAULT_FLOAT_DTYPE_TF
    )


def test_straight_through_categorical_forward_is_one_hot():
    logits = tf.constant([0.5, -1.0, 2.0, 0.0], dtype=DEFAULT_FLOAT_DTYPE_TF)
    one_hot, index = straight_through_categorical(logits, seed=1)
    values = one_hot.numpy()
    assert set(np.unique(values)).issubset({0.0, 1.0})
    assert np.isclose(values.sum(), 1.0)
    assert values[int(index)] == 1.0


def test_straight_through_categorical_gradient_is_softmax_jacobian():
    logits = tf.Variable([0.5, -1.0, 2.0], dtype=DEFAULT_FLOAT_DTYPE_TF)
    with tf.GradientTape() as tape:
        one_hot, _ = straight_through_categorical(logits, seed=2)
        # sum of the straight-through one-hot
        total = tf.reduce_sum(one_hot)
    grad = tape.gradient(total, logits)
    # gradient flows as through softmax: d sum(softmax)/d logits == 0
    assert np.allclose(grad.numpy(), 0.0, atol=1e-8)


def test_gumbel_softmax_hard_forward_one_hot():
    logits = tf.constant([1.0, 0.0, -1.0], dtype=DEFAULT_FLOAT_DTYPE_TF)
    sample, index = gumbel_softmax(logits, temperature=0.5, hard=True, seed=3)
    values = sample.numpy()
    assert set(np.unique(values)).issubset({0.0, 1.0})
    assert values[int(index)] == 1.0


def test_gumbel_softmax_gradient_flows():
    logits = tf.Variable([1.0, 0.0, -1.0], dtype=DEFAULT_FLOAT_DTYPE_TF)
    with tf.GradientTape() as tape:
        sample, _ = gumbel_softmax(logits, temperature=0.5, hard=True, seed=4)
        loss = sample[0]
    grad = tape.gradient(loss, logits)
    assert grad is not None
    assert np.any(grad.numpy() != 0)


def test_leave_one_out_baseline():
    values = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=DEFAULT_FLOAT_DTYPE_TF)
    baseline = leave_one_out_baseline(values).numpy()
    # baseline_0 = mean(2,3,4) = 3, baseline_3 = mean(1,2,3) = 2
    assert np.isclose(baseline[0], 3.0)
    assert np.isclose(baseline[3], 2.0)


def test_score_function_surrogate_unbiased_against_exact():
    """REINFORCE gradient should match the exact gradient in expectation."""
    n = 4
    support = ConditionalCladeSupport(n)
    target_logits = random_logits(support, seed=11)
    p = ConditionalCladeDistribution(support, target_logits)

    q_logits = tf.Variable(random_logits(support, seed=12))
    q = ConditionalCladeDistribution(support, q_logits)

    # Exact gradient of KL(q || p) w.r.t. q logits.
    with tf.GradientTape() as tape:
        exact_kl = q.exact_kl_divergence(p)
    exact_grad = tape.gradient(exact_kl, q_logits).numpy()

    # Monte-Carlo REINFORCE estimate, averaged over many batches.
    rng = np.random.default_rng(20)
    log_p_cond = p.conditional_log_probs()
    n_batches = 400
    batch_size = 32
    grad_accum = np.zeros_like(exact_grad)
    for _ in range(n_batches):
        flat_indices = q.sample_flat_index_batch(batch_size, rng)
        flat_tensor = tf.constant(flat_indices, tf.int32)
        with tf.GradientTape() as tape:
            log_q = q.log_prob_from_flat_indices(flat_tensor)
            log_p = tf.reduce_sum(tf.gather(log_p_cond, flat_tensor), axis=-1)
            cost = log_q - log_p  # per-sample reverse-KL integrand
            baseline = leave_one_out_baseline(cost)
            surrogate = score_function_surrogate(cost, log_q, baseline)
        grad_accum += tape.gradient(surrogate, q_logits).numpy()
    mc_grad = grad_accum / n_batches

    # Compare on a cosine-similarity / relative basis (MC is noisy).
    cosine = np.dot(mc_grad, exact_grad) / (
        np.linalg.norm(mc_grad) * np.linalg.norm(exact_grad)
    )
    assert cosine > 0.95


def test_vimco_surrogate_drives_q_toward_p():
    """A few VIMCO ascent steps should reduce KL(q || p)."""
    n = 4
    support = ConditionalCladeSupport(n)
    p = ConditionalCladeDistribution(support, random_logits(support, seed=30))
    q_logits = tf.Variable(random_logits(support, seed=31))
    q = ConditionalCladeDistribution(support, q_logits)
    log_p_cond = p.conditional_log_probs()

    optimizer = tf.optimizers.Adam(0.2)
    rng = np.random.default_rng(40)
    initial_kl = q.exact_kl_divergence(p).numpy()
    for _ in range(150):
        flat_indices = q.sample_flat_index_batch(16, rng)
        flat_tensor = tf.constant(flat_indices, tf.int32)
        with tf.GradientTape() as tape:
            log_q = q.log_prob_from_flat_indices(flat_tensor)
            log_p = tf.reduce_sum(tf.gather(log_p_cond, flat_tensor), axis=-1)
            log_weights = log_p - log_q
            loss = -vimco_surrogate(log_q, log_weights)  # maximise bound
        grads = tape.gradient(loss, [q_logits])
        optimizer.apply_gradients(zip(grads, [q_logits]))
    final_kl = q.exact_kl_divergence(p).numpy()
    assert final_kl < initial_kl


def test_relaxed_cost_is_differentiable():
    n = 4
    support = ConditionalCladeSupport(n)
    p = ConditionalCladeDistribution(support, random_logits(support, seed=50))
    q_logits = tf.Variable(random_logits(support, seed=51))
    q = ConditionalCladeDistribution(support, q_logits)
    tf.random.set_seed(7)
    with tf.GradientTape() as tape:
        sample = sample_relaxed_cost(q, p, temperature=0.5, gumbel=True)
        cost = sample.log_q - sample.log_p
    grad = tape.gradient(cost, q_logits)
    assert grad is not None
    assert np.any(grad.numpy() != 0)
    # the realised topology is a valid parent_indices array
    assert sample.parent_indices.shape == (2 * n - 2,)


def test_straight_through_relaxation_reduces_kl():
    n = 4
    support = ConditionalCladeSupport(n)
    p = ConditionalCladeDistribution(support, random_logits(support, seed=60))
    q_logits = tf.Variable(random_logits(support, seed=61))
    q = ConditionalCladeDistribution(support, q_logits)

    optimizer = tf.optimizers.Adam(0.1)
    initial_kl = q.exact_kl_divergence(p).numpy()
    tf.random.set_seed(123)
    for _ in range(300):
        with tf.GradientTape() as tape:
            costs = []
            for _ in range(8):
                sample = sample_relaxed_cost(q, p, temperature=0.5, gumbel=True)
                costs.append(sample.log_q - sample.log_p)
            loss = tf.add_n(costs) / len(costs)
        grads = tape.gradient(loss, [q_logits])
        optimizer.apply_gradients(zip(grads, [q_logits]))
    final_kl = q.exact_kl_divergence(p).numpy()
    # Straight-through is biased but should still make clear progress.
    assert final_kl < initial_kl
