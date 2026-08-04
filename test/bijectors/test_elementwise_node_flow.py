"""Tests for the elementwise (nonlinear) layer of the tree normalising flow."""
import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.bijectors.elementwise_node_flow import (
    DEFAULT_BOUNDS,
    NONLINEARITIES,
    ElementwiseNodeFlow,
    build_nonlinearity,
    identity_parameters,
    shared_parameter_shapes,
)

NODE_COUNT = 6
NUM_BINS = 8


def constant(x):
    return tf.constant(np.asarray(x), dtype=DEFAULT_FLOAT_DTYPE_TF)


@pytest.fixture
def affine_parameters():
    rng = np.random.default_rng(0)
    return (
        constant(rng.uniform(0.5, 1.5, size=(NODE_COUNT,))),
        constant(rng.normal(size=(NODE_COUNT,))),
    )


def random_parameters(nonlinearity, seed=1, batch_shape=()):
    rng = np.random.default_rng(seed)
    return {
        name: constant(rng.normal(size=tuple(batch_shape) + shape))
        for name, shape in shared_parameter_shapes(nonlinearity, NUM_BINS).items()
    }


def build(nonlinearity, affine_parameters, parameters=None, seed=1):
    if parameters is None:
        parameters = random_parameters(nonlinearity, seed=seed)
    return ElementwiseNodeFlow(
        *affine_parameters,
        build_nonlinearity(nonlinearity, parameters),
    )


@pytest.fixture
def x():
    return constant(np.random.default_rng(2).normal(size=(4, NODE_COUNT)))


@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
def test_identity_parameters_leave_only_the_affine(
    nonlinearity, affine_parameters, x
):
    """At zero raw parameters every nonlinearity is the identity, so the layer
    is exactly its per-node affine conditioning."""
    scale, shift = affine_parameters
    bijector = build(
        nonlinearity,
        affine_parameters,
        identity_parameters(
            nonlinearity, NUM_BINS, dtype=DEFAULT_FLOAT_DTYPE_TF
        ),
    )
    assert_allclose(
        bijector.forward(x).numpy(), (scale * x + shift).numpy(), rtol=1e-8, atol=1e-8
    )
    assert_allclose(
        bijector.forward_log_det_jacobian(x, event_ndims=1).numpy(),
        np.full(x.shape[0], np.log(scale.numpy()).sum()),
        rtol=1e-8,
        atol=1e-8,
    )


@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
def test_roundtrip(nonlinearity, affine_parameters, x):
    bijector = build(nonlinearity, affine_parameters)
    y = bijector.forward(x)
    assert_allclose(bijector.inverse(y).numpy(), x.numpy(), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
def test_log_det_jacobians_agree(nonlinearity, affine_parameters, x):
    bijector = build(nonlinearity, affine_parameters)
    y = bijector.forward(x)
    assert_allclose(
        bijector.forward_log_det_jacobian(x, event_ndims=1).numpy(),
        -bijector.inverse_log_det_jacobian(y, event_ndims=1).numpy(),
        rtol=1e-9,
    )


@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
def test_log_det_jacobian_matches_determinant(nonlinearity, affine_parameters):
    bijector = build(nonlinearity, affine_parameters)
    x = constant(np.random.default_rng(3).normal(size=(NODE_COUNT,)))
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = bijector.forward(x)
    jacobian = tape.jacobian(y, x).numpy()
    # Elementwise: the Jacobian must be diagonal.
    assert_allclose(jacobian - np.diag(np.diag(jacobian)), 0.0, atol=1e-14)
    sign, log_abs_det = np.linalg.slogdet(jacobian)
    assert sign > 0
    assert_allclose(
        log_abs_det,
        bijector.forward_log_det_jacobian(x, event_ndims=1).numpy(),
        rtol=1e-9,
    )


@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
def test_monotone(nonlinearity, affine_parameters):
    bijector = build(nonlinearity, affine_parameters)
    grid = constant(
        np.linspace(-4 * DEFAULT_BOUNDS, 4 * DEFAULT_BOUNDS, 400)[:, None]
        * np.ones((1, NODE_COUNT))
    )
    assert (np.diff(bijector.forward(grid).numpy(), axis=0) > 0).all()


def test_spline_has_identity_tails(affine_parameters):
    """Outside its range the spline is the identity, so the layer's tails stay
    linear (Gaussian in flow space, and so logit-normal once the ratio
    bijector's sigmoid is applied downstream) however the knots move."""
    scale, shift = affine_parameters
    bijector = build("spline", affine_parameters)
    far_out = constant(np.full((1, NODE_COUNT), 10 * DEFAULT_BOUNDS))
    assert_allclose(
        bijector.forward(far_out).numpy(),
        (scale * far_out + shift).numpy(),
        rtol=1e-10,
    )


def test_sinh_arcsinh_reshapes_the_tails(affine_parameters):
    """The unbounded alternative: unlike the spline it does change the tails,
    which is the reason to reach for it."""
    scale, shift = affine_parameters
    bijector = build("sinh_arcsinh", affine_parameters)
    far_out = constant(np.full((1, NODE_COUNT), 10 * DEFAULT_BOUNDS))
    affine = (scale * far_out + shift).numpy()
    assert (np.abs(bijector.forward(far_out).numpy() - affine) > 1.0).all()


def test_affine_nonlinearity_is_linear(affine_parameters):
    """``"affine"`` really is linear: its Jacobian does not vary with x."""
    bijector = build("affine", affine_parameters)
    jacobians = []
    for value in (-2.0, 2.0):
        x = constant(np.full(NODE_COUNT, value))
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = bijector.forward(x)
        jacobians.append(tape.jacobian(y, x).numpy())
    assert_allclose(jacobians[0], jacobians[1], rtol=1e-12)


@pytest.mark.parametrize("nonlinearity", ["spline", "sinh_arcsinh"])
def test_nonlinearity_is_nonlinear(nonlinearity, affine_parameters):
    jacobians = []
    for value in (-1.0, 1.0):
        x = constant(np.full(NODE_COUNT, value))
        bijector = build(nonlinearity, affine_parameters)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = bijector.forward(x)
        jacobians.append(tape.jacobian(y, x).numpy())
    assert np.abs(jacobians[0] - jacobians[1]).max() > 1e-6


def test_per_node_parameters_broadcast(affine_parameters, x):
    """Per-node knots are supported (they simply broadcast), even though the
    flow shares one nonlinearity across the tree by default."""
    bijector = build(
        "spline",
        affine_parameters,
        random_parameters("spline", seed=4, batch_shape=(NODE_COUNT,)),
    )
    y = bijector.forward(x)
    assert tuple(y.shape) == tuple(x.shape)
    assert_allclose(bijector.inverse(y).numpy(), x.numpy(), rtol=1e-10, atol=1e-10)


def test_invalid_nonlinearity(affine_parameters):
    with pytest.raises(ValueError, match="nonlinearity must be"):
        shared_parameter_shapes("quadratic")


@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
def test_gradients_reach_all_parameters(nonlinearity, x):
    """The shared nonlinearity's parameters and the per-node affine are all
    trainable."""
    affine_variables = [
        tf.Variable(tf.fill([NODE_COUNT], tf.constant(0.5, DEFAULT_FLOAT_DTYPE_TF))),
        tf.Variable(tf.zeros([NODE_COUNT], DEFAULT_FLOAT_DTYPE_TF)),
    ]
    shared_variables = {
        name: tf.Variable(tf.zeros(shape, DEFAULT_FLOAT_DTYPE_TF))
        for name, shape in shared_parameter_shapes(nonlinearity, NUM_BINS).items()
    }
    with tf.GradientTape() as tape:
        bijector = ElementwiseNodeFlow(
            tf.math.softplus(affine_variables[0]),
            affine_variables[1],
            build_nonlinearity(nonlinearity, shared_variables),
        )
        loss = tf.reduce_sum(bijector.forward(x)) + tf.reduce_sum(
            bijector.forward_log_det_jacobian(x, event_ndims=1)
        )
    variables = affine_variables + list(shared_variables.values())
    gradients = tape.gradient(loss, variables)
    assert all(gradient is not None for gradient in gradients)
    assert all(np.isfinite(gradient.numpy()).all() for gradient in gradients)
