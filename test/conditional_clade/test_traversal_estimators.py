"""Tests for the vectorised pre-sampled-traversal gradient helpers."""

import numpy as np
import pytest
import tensorflow as tf

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.conditional_clade.distribution import ConditionalCladeDistribution
from treeflow.conditional_clade.estimators import sample_relaxed_cost
from treeflow.conditional_clade.support import ConditionalCladeSupport
from treeflow.conditional_clade.traversal_estimators import (
    straight_through_traversal_cost,
    straight_through_traversal_log_prob,
    traversal_log_prob,
)


def random_logits(support, seed=0):
    rng = np.random.default_rng(seed)
    return tf.constant(
        rng.standard_normal(support.subsplit_count), dtype=DEFAULT_FLOAT_DTYPE_TF
    )


def make(n, seed=0):
    support = ConditionalCladeSupport(n)
    logits = tf.Variable(random_logits(support, seed))
    return support, logits, ConditionalCladeDistribution(support, logits)


@pytest.mark.parametrize("gumbel", [False, True])
def test_straight_through_forward_equals_exact(gumbel):
    support, logits, q = make(6, seed=1)
    flat = tf.constant(
        q.sample_flat_index_batch(25, np.random.default_rng(2)), tf.int32
    )
    exact = traversal_log_prob(q.conditional_log_probs(), flat).numpy()
    st = straight_through_traversal_log_prob(
        q.conditional_log_probs(),
        tf.convert_to_tensor(q.logits, DEFAULT_FLOAT_DTYPE_TF),
        flat,
        support.segment_ids,
        temperature=0.5,
        gumbel=gumbel,
        seed=(1, 2),
    ).numpy()
    # Forward value is the exact log-prob regardless of the relaxation.
    np.testing.assert_allclose(st, exact, atol=1e-10)


def test_traversal_log_prob_matches_distribution():
    support, logits, q = make(5, seed=3)
    flat = tf.constant(
        q.sample_flat_index_batch(16, np.random.default_rng(4)), tf.int32
    )
    np.testing.assert_allclose(
        traversal_log_prob(q.conditional_log_probs(), flat).numpy(),
        q.log_prob_from_flat_indices(flat).numpy(),
    )


def test_straight_through_gradient_is_finite_and_nonzero():
    support, logits, q = make(6, seed=5)
    flat = tf.constant(
        q.sample_flat_index_batch(32, np.random.default_rng(6)), tf.int32
    )
    with tf.GradientTape() as tape:
        value = tf.reduce_sum(
            q.straight_through_log_prob_from_flat_indices(
                flat, temperature=0.5, gumbel=True, seed=(7, 8)
            )
        )
    grad = tape.gradient(value, logits).numpy()
    assert np.all(np.isfinite(grad))
    assert np.any(grad != 0)


def test_straight_through_temperature1_is_categorical_relaxation():
    """At temperature 1 with no Gumbel noise the relaxation is the per-clade
    softmax: the gradient equals that of the expected conditional log-prob."""
    support, logits, q = make(5, seed=9)
    flat = tf.constant(
        q.sample_flat_index_batch(8, np.random.default_rng(10)), tf.int32
    )
    with tf.GradientTape() as tape:
        st = tf.reduce_sum(
            q.straight_through_log_prob_from_flat_indices(flat, temperature=1.0)
        )
    st_grad = tape.gradient(st, logits)

    # Reference: sum over the visited clades of E_{q(.|clade)}[cond] (the soft
    # term the straight-through gradient flows through).
    seg_ids = support.segment_ids
    with tf.GradientTape() as tape:
        cond = q.conditional_log_probs()
        probs = tf.exp(cond)
        seg_expected = tf.math.unsorted_segment_sum(
            probs * cond, seg_ids, support.parent_clade_count
        )
        seg_of = tf.gather(seg_ids, tf.reshape(flat, [-1]))
        ref = tf.reduce_sum(tf.gather(seg_expected, seg_of))
    ref_grad = tape.gradient(ref, logits)
    np.testing.assert_allclose(st_grad.numpy(), ref_grad.numpy(), atol=1e-10)


def test_cost_forward_equals_exact_reverse_kl_integrand():
    support, logits, q = make(6, seed=11)
    p = ConditionalCladeDistribution(support, random_logits(support, seed=12))
    flat = tf.constant(
        q.sample_flat_index_batch(20, np.random.default_rng(13)), tf.int32
    )
    cost = straight_through_traversal_cost(
        q.conditional_log_probs(),
        p.conditional_log_probs(),
        tf.convert_to_tensor(q.logits, DEFAULT_FLOAT_DTYPE_TF),
        flat,
        support.segment_ids,
        temperature=0.5,
        gumbel=True,
        seed=(1, 1),
    ).numpy()
    exact = (
        q.log_prob_from_flat_indices(flat) - p.log_prob_from_flat_indices(flat)
    ).numpy()
    np.testing.assert_allclose(cost, exact, atol=1e-10)


def test_cost_gradient_flows_through_q_for_both_terms():
    """The target term log p must still carry a gradient w.r.t. q (through the
    shared relaxation), as in the recursive reference."""
    support, logits, q = make(5, seed=14)
    p = ConditionalCladeDistribution(support, random_logits(support, seed=15))
    flat = tf.constant(
        q.sample_flat_index_batch(16, np.random.default_rng(16)), tf.int32
    )
    pc = p.conditional_log_probs()
    with tf.GradientTape() as tape:
        # only the log p term (q's cond replaced by a constant), to isolate that
        # the relaxation still routes a gradient into q from the target term
        weights_logits = tf.convert_to_tensor(q.logits, DEFAULT_FLOAT_DTYPE_TF)
        log_q_const = tf.stop_gradient(q.conditional_log_probs())
        cost = straight_through_traversal_cost(
            log_q_const, pc, weights_logits, flat, support.segment_ids,
            temperature=0.5, gumbel=False,
        )
        loss = tf.reduce_sum(cost)
    grad = tape.gradient(loss, logits)
    assert grad is not None and np.any(grad.numpy() != 0)


def test_vectorized_cost_trains_reduces_kl():
    support = ConditionalCladeSupport(5)
    p = ConditionalCladeDistribution(support, random_logits(support, seed=20))
    logits = tf.Variable(random_logits(support, seed=21))
    q = ConditionalCladeDistribution(support, logits)
    pc = p.conditional_log_probs()
    seg = tf.constant(support.segment_ids)
    opt = tf.optimizers.Adam(0.05)
    rng = np.random.default_rng(100)
    initial = float(q.exact_kl_divergence(p))

    @tf.function
    def step(flat):
        with tf.GradientTape() as tape:
            cost = straight_through_traversal_cost(
                q.conditional_log_probs(), pc,
                tf.convert_to_tensor(q.logits, DEFAULT_FLOAT_DTYPE_TF),
                flat, seg, temperature=0.5, gumbel=True,
            )
            loss = tf.reduce_mean(cost)
        opt.apply_gradients([(tape.gradient(loss, logits), logits)])

    for _ in range(400):
        step(tf.constant(q.sample_flat_index_batch(32, rng), tf.int32))
    assert float(q.exact_kl_divergence(p)) < initial - 0.5


def test_matches_recursive_reference_forward():
    """The vectorised cost's forward value matches the recursive reference for
    the same realised topology (both equal the exact reverse-KL integrand)."""
    support, logits, q = make(5, seed=30)
    p = ConditionalCladeDistribution(support, random_logits(support, seed=31))
    tf.random.set_seed(0)
    sample = sample_relaxed_cost(q, p, temperature=0.5, gumbel=True)
    flat = tf.constant(
        support.assignment_flat_indices(
            support.parent_indices_to_assignment(sample.parent_indices)
        )[np.newaxis],
        tf.int32,
    )
    recursive_cost = float(sample.log_q - sample.log_p)
    vectorized_cost = float(
        straight_through_traversal_cost(
            q.conditional_log_probs(), p.conditional_log_probs(),
            tf.convert_to_tensor(q.logits, DEFAULT_FLOAT_DTYPE_TF),
            flat, support.segment_ids, temperature=0.5, gumbel=True,
        )[0]
    )
    assert np.isclose(recursive_cost, vectorized_cost)
