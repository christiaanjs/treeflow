"""Correctness tests for the native node-height ratio transform custom op.

Everything is checked against the reference TensorFlow implementation in
``treeflow.traversal.ratio_transform`` (forward values and autodiff gradients),
the analytic gradient is spot-checked against finite differences, and the
integration through ``NodeHeightRatioBijector`` is exercised end to end.
"""
import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

from treeflow.acceleration.native import native_ratios_to_node_heights
from treeflow.traversal.ratio_transform import ratios_to_node_heights as reference


def _args(problem):
    return (
        problem["preorder_node_indices"],
        problem["parent_indices"],
        problem["ratios"],
        problem["anchor_heights"],
    )


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function_mode", [False, True])
def test_native_matches_reference_forward(small_ratio_problem, function_mode):
    ref_fn = tf.function(reference) if function_mode else reference
    nat_fn = (
        tf.function(native_ratios_to_node_heights)
        if function_mode
        else native_ratios_to_node_heights
    )
    p = small_ratio_problem
    ref_val = ref_fn(p["topology"], p["ratios"], p["anchor_heights"])
    nat_val = nat_fn(*_args(p))
    assert_allclose(nat_val.numpy(), ref_val.numpy(), rtol=1e-12, atol=1e-12)


def test_native_matches_known_transform():
    """Reproduce the worked example from the ratio-bijector fixtures."""
    parent_indices = np.array([5, 5, 6, 6, 8, 7, 7, 8])
    taxon_count = 5
    preorder = tf.constant(np.array([8, 7, 5, 6]) - taxon_count, tf.int32)
    parent = tf.constant(parent_indices[taxon_count:] - taxon_count, tf.int32)
    ratios = tf.constant([0.25, 0.625, 0.5, 1.6], tf.float64)
    anchor = tf.zeros(4, tf.float64)
    heights = native_ratios_to_node_heights(preorder, parent, ratios, anchor)
    assert_allclose(heights.numpy(), [0.2, 0.5, 0.8, 1.6])


def test_native_float32(small_ratio_problem):
    p = small_ratio_problem
    ratios = tf.cast(p["ratios"], tf.float32)
    anchor = tf.cast(p["anchor_heights"], tf.float32)
    nat = native_ratios_to_node_heights(
        p["preorder_node_indices"], p["parent_indices"], ratios, anchor
    )
    ref = reference(p["topology"], ratios, anchor)
    assert nat.dtype == tf.float32
    assert_allclose(nat.numpy(), ref.numpy(), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient correctness (analytic native gradient vs reference autodiff)
# ---------------------------------------------------------------------------


def _grads(fn, problem, native=True):
    ratios = tf.Variable(problem["ratios"])
    anchor = tf.Variable(problem["anchor_heights"])
    # Non-uniform weights so every node-height adjoint is distinct.
    weights = tf.reshape(
        tf.range(1, problem["node_count"] + 1, dtype=problem["ratios"].dtype),
        tf.shape(problem["ratios"])[-1:],
    )
    with tf.GradientTape() as tape:
        if native:
            heights = fn(
                problem["preorder_node_indices"],
                problem["parent_indices"],
                ratios,
                anchor,
            )
        else:
            heights = fn(problem["topology"], ratios, anchor)
        loss = tf.reduce_sum(heights * weights)
    return tape.gradient(loss, [ratios, anchor])


def test_native_gradient_matches_reference(small_ratio_problem):
    ref_grads = _grads(reference, small_ratio_problem, native=False)
    nat_grads = _grads(native_ratios_to_node_heights, small_ratio_problem, native=True)
    for ref_g, nat_g in zip(ref_grads, nat_grads):
        assert nat_g is not None
        assert_allclose(nat_g.numpy(), ref_g.numpy(), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("function_mode", [False, True])
def test_native_gradient_function_mode(small_ratio_problem, function_mode):
    fn = (
        tf.function(native_ratios_to_node_heights)
        if function_mode
        else native_ratios_to_node_heights
    )
    ref_grads = _grads(reference, small_ratio_problem, native=False)
    nat_grads = _grads(fn, small_ratio_problem, native=True)
    for ref_g, nat_g in zip(ref_grads, nat_grads):
        assert_allclose(nat_g.numpy(), ref_g.numpy(), rtol=1e-9, atol=1e-9)


def test_native_gradient_finite_difference(small_ratio_problem):
    """Spot-check the analytic gradient against central finite differences."""
    p = small_ratio_problem
    ratios0 = p["ratios"]
    weights = tf.range(1, p["node_count"] + 1, dtype=ratios0.dtype)

    def loss_of(ratios):
        heights = native_ratios_to_node_heights(
            p["preorder_node_indices"],
            p["parent_indices"],
            ratios,
            p["anchor_heights"],
        )
        return tf.reduce_sum(heights * weights)

    ratios_var = tf.Variable(ratios0)
    with tf.GradientTape() as tape:
        loss = loss_of(ratios_var)
    analytic = tape.gradient(loss, ratios_var).numpy()

    eps = 1e-6
    flat = ratios0.numpy().reshape(-1).copy()
    for idx in range(flat.size):
        plus = flat.copy()
        plus[idx] += eps
        minus = flat.copy()
        minus[idx] -= eps
        lp = loss_of(tf.constant(plus.reshape(ratios0.shape))).numpy()
        lm = loss_of(tf.constant(minus.reshape(ratios0.shape))).numpy()
        fd = (lp - lm) / (2 * eps)
        assert_allclose(analytic.reshape(-1)[idx], fd, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# Batching (leading sample/site dims over the ratios)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_shape", [(5,), (3, 4)])
def test_native_batched_matches_reference(
    make_ratio_problem_factory, batch_shape
):
    p = make_ratio_problem_factory(leaf_count=10, batch_shape=batch_shape, seed=9)
    nat = native_ratios_to_node_heights(*_args(p))
    ref = reference(p["topology"], p["ratios"], p["anchor_heights"])
    assert tuple(nat.shape) == batch_shape + (p["node_count"],)
    assert_allclose(nat.numpy(), ref.numpy(), rtol=1e-12, atol=1e-12)


def test_native_batched_gradient_matches_reference(make_ratio_problem_factory):
    p = make_ratio_problem_factory(leaf_count=10, batch_shape=(4,), seed=12)
    ref_grads = _grads(reference, p, native=False)
    nat_grads = _grads(native_ratios_to_node_heights, p, native=True)
    for ref_g, nat_g in zip(ref_grads, nat_grads):
        assert nat_g is not None
        assert nat_g.shape == ref_g.shape
        assert_allclose(nat_g.numpy(), ref_g.numpy(), rtol=1e-9, atol=1e-9)


def test_native_broadcast_anchor_gradient_reduces(make_ratio_problem_factory):
    """A single (unbatched) anchor shared across a ratio batch: its gradient
    must reduce back to the anchor's own ``[node]`` shape."""
    p = make_ratio_problem_factory(leaf_count=9, batch_shape=(6,), seed=14)
    anchor = tf.Variable(p["anchor_heights"])  # shape [node]
    ratios = tf.Variable(p["ratios"])  # shape [6, node]
    with tf.GradientTape() as tape:
        nat = native_ratios_to_node_heights(
            p["preorder_node_indices"], p["parent_indices"], ratios, anchor
        )
        loss = tf.reduce_sum(nat)
    with tf.GradientTape() as tape_ref:
        ref = reference(p["topology"], ratios, anchor)
        loss_ref = tf.reduce_sum(ref)
    nat_g = tape.gradient(loss, anchor)
    ref_g = tape_ref.gradient(loss_ref, anchor)
    assert tuple(nat_g.shape) == (p["node_count"],)
    assert_allclose(nat_g.numpy(), ref_g.numpy(), rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Integration through NodeHeightRatioBijector
# ---------------------------------------------------------------------------


def _bijector(problem, use_native):
    from treeflow.bijectors.node_height_ratio_bijector import (
        NodeHeightRatioBijector,
    )

    return NodeHeightRatioBijector(
        problem["topology"], problem["anchor_heights"], use_native=use_native
    )


def test_bijector_native_matches_default_forward(small_ratio_problem):
    p = small_ratio_problem
    native = _bijector(p, use_native=True).forward(p["ratios"])
    default = _bijector(p, use_native=False).forward(p["ratios"])
    assert_allclose(native.numpy(), default.numpy(), rtol=1e-12, atol=1e-12)


def test_bijector_auto_uses_native_when_available(small_ratio_problem):
    bijector = _bijector(small_ratio_problem, use_native="auto")
    # The native op is built for this package (see conftest), so auto resolves
    # to using it.
    assert bijector._use_native is True


def test_bijector_invalid_use_native(small_ratio_problem):
    with pytest.raises(ValueError):
        _bijector(small_ratio_problem, use_native="sometimes")


def test_bijector_native_forward_log_det_jacobian_gradient(small_ratio_problem):
    """The reparameterisation term used by inference (forward + fldj) and its
    first-order gradient match the pure-TensorFlow bijector."""
    p = small_ratio_problem
    native_bijector = _bijector(p, use_native=True)
    tf_bijector = _bijector(p, use_native=False)

    def value_and_grad(bijector):
        ratios = tf.Variable(p["ratios"])
        with tf.GradientTape() as tape:
            value = tf.reduce_sum(
                bijector.forward(ratios)
            ) + bijector.forward_log_det_jacobian(ratios, event_ndims=1)
        return value.numpy(), tape.gradient(value, ratios).numpy()

    nat_v, nat_g = value_and_grad(native_bijector)
    tf_v, tf_g = value_and_grad(tf_bijector)
    assert_allclose(nat_v, tf_v, rtol=1e-12, atol=1e-12)
    assert_allclose(nat_g, tf_g, rtol=1e-9, atol=1e-9)


def test_bijector_native_roundtrip(small_ratio_problem):
    """Inverse (pure TF) composed with the native forward is the identity."""
    p = small_ratio_problem
    bijector = _bijector(p, use_native=True)
    heights = bijector.forward(p["ratios"])
    recovered = bijector.inverse(heights)
    assert_allclose(recovered.numpy(), p["ratios"].numpy(), rtol=1e-9, atol=1e-9)
