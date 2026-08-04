"""Correctness tests for the native affine tree map custom ops.

Everything is checked against the reference TensorFlow implementations in
``treeflow.traversal.tree_affine`` (forward values and autodiff gradients), the
analytic gradients are spot-checked against finite differences, and the
integration through the affine bijectors and the tree normalising flow is
exercised end to end.
"""
import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

from treeflow.acceleration.native import (
    native_postorder_affine,
    native_preorder_affine,
)
from treeflow.traversal import tree_affine as reference
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
from treeflow.tree.topology.tensorflow_tree_topology import numpy_topology_to_tensor


def _random_parent_indices(leaf_count: int, rng: np.random.Generator) -> np.ndarray:
    node_count = 2 * leaf_count - 1
    parent = np.full(node_count, -1, dtype=np.int32)
    active = list(range(leaf_count))
    next_node = leaf_count
    while len(active) > 1:
        a = active.pop(rng.integers(len(active)))
        b = active.pop(rng.integers(len(active)))
        parent[a] = next_node
        parent[b] = next_node
        active.append(next_node)
        next_node += 1
    return parent[:-1]


def make_affine_problem(leaf_count=9, batch_shape=(), seed=0, dtype=tf.float64):
    """A random affine tree map problem: topology, coordinates and parameters."""
    rng = np.random.default_rng(seed)
    topology = numpy_topology_to_tensor(
        NumpyTreeTopology(parent_indices=_random_parent_indices(leaf_count, rng))
    )
    node_count = leaf_count - 1
    np_dtype = dtype.as_numpy_dtype
    shape = tuple(batch_shape) + (node_count,)

    def constant(x):
        return tf.constant(np.asarray(x, dtype=np_dtype))

    return dict(
        topology=topology,
        leaf_count=leaf_count,
        node_count=node_count,
        preorder_node_indices=topology.preorder_node_indices - leaf_count,
        postorder_node_indices=topology.postorder_node_indices - leaf_count,
        parent_indices=reference.node_parent_indices(topology),
        child_indices=reference.node_child_indices(topology),
        x=constant(rng.normal(size=shape)),
        scale=constant(rng.uniform(0.5, 1.5, size=(node_count,))),
        shift=constant(rng.normal(size=(node_count,))),
        parent_weight=constant(rng.uniform(-0.9, 0.9, size=(node_count,))),
        child_weight=constant(rng.uniform(-0.4, 0.4, size=(node_count, 2))),
    )


@pytest.fixture
def affine_problem():
    return make_affine_problem(seed=5)


def _native_preorder(problem, x=None, scale=None, shift=None, weight=None):
    return native_preorder_affine(
        problem["preorder_node_indices"],
        problem["parent_indices"],
        problem["x"] if x is None else x,
        problem["scale"] if scale is None else scale,
        problem["shift"] if shift is None else shift,
        problem["parent_weight"] if weight is None else weight,
    )


def _native_postorder(problem, x=None, scale=None, shift=None, weight=None):
    return native_postorder_affine(
        problem["postorder_node_indices"],
        problem["child_indices"],
        problem["x"] if x is None else x,
        problem["scale"] if scale is None else scale,
        problem["shift"] if shift is None else shift,
        problem["child_weight"] if weight is None else weight,
    )


def _reference_preorder(problem, **kwargs):
    return reference.preorder_affine(
        problem["topology"],
        kwargs.get("x", problem["x"]),
        kwargs.get("scale", problem["scale"]),
        kwargs.get("shift", problem["shift"]),
        kwargs.get("weight", problem["parent_weight"]),
    )


def _reference_postorder(problem, **kwargs):
    return reference.postorder_affine(
        problem["topology"],
        kwargs.get("x", problem["x"]),
        kwargs.get("scale", problem["scale"]),
        kwargs.get("shift", problem["shift"]),
        kwargs.get("weight", problem["child_weight"]),
    )


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function_mode", [False, True])
def test_native_preorder_matches_reference(affine_problem, function_mode):
    native = (
        tf.function(native_preorder_affine) if function_mode else native_preorder_affine
    )
    value = native(
        affine_problem["preorder_node_indices"],
        affine_problem["parent_indices"],
        affine_problem["x"],
        affine_problem["scale"],
        affine_problem["shift"],
        affine_problem["parent_weight"],
    )
    assert_allclose(
        value.numpy(), _reference_preorder(affine_problem).numpy(), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("function_mode", [False, True])
def test_native_postorder_matches_reference(affine_problem, function_mode):
    native = (
        tf.function(native_postorder_affine)
        if function_mode
        else native_postorder_affine
    )
    value = native(
        affine_problem["postorder_node_indices"],
        affine_problem["child_indices"],
        affine_problem["x"],
        affine_problem["scale"],
        affine_problem["shift"],
        affine_problem["child_weight"],
    )
    assert_allclose(
        value.numpy(),
        _reference_postorder(affine_problem).numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_native_float32(affine_problem):
    cast = lambda x: tf.cast(x, tf.float32)
    native = _native_preorder(
        affine_problem,
        x=cast(affine_problem["x"]),
        scale=cast(affine_problem["scale"]),
        shift=cast(affine_problem["shift"]),
        weight=cast(affine_problem["parent_weight"]),
    )
    ref = _reference_preorder(
        affine_problem,
        x=cast(affine_problem["x"]),
        scale=cast(affine_problem["scale"]),
        shift=cast(affine_problem["shift"]),
        weight=cast(affine_problem["parent_weight"]),
    )
    assert native.dtype == tf.float32
    assert_allclose(native.numpy(), ref.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("batch_shape", [(5,), (3, 4)])
def test_native_batched_matches_reference(batch_shape):
    problem = make_affine_problem(leaf_count=10, batch_shape=batch_shape, seed=9)
    for native, ref in (
        (_native_preorder(problem), _reference_preorder(problem)),
        (_native_postorder(problem), _reference_postorder(problem)),
    ):
        assert tuple(native.shape) == batch_shape + (problem["node_count"],)
        assert_allclose(native.numpy(), ref.numpy(), rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Gradient correctness
# ---------------------------------------------------------------------------


def _grads(problem, native, postorder):
    x = tf.Variable(problem["x"])
    scale = tf.Variable(problem["scale"])
    shift = tf.Variable(problem["shift"])
    weight = tf.Variable(
        problem["child_weight"] if postorder else problem["parent_weight"]
    )
    # Non-uniform weights so every node's adjoint is distinct.
    loss_weights = tf.reshape(
        tf.range(1, problem["node_count"] + 1, dtype=problem["x"].dtype),
        [problem["node_count"]],
    )
    with tf.GradientTape() as tape:
        if postorder:
            fn = _native_postorder if native else _reference_postorder
        else:
            fn = _native_preorder if native else _reference_preorder
        value = fn(problem, x=x, scale=scale, shift=shift, weight=weight)
        loss = tf.reduce_sum(value * loss_weights)
    return tape.gradient(loss, [x, scale, shift, weight])


@pytest.mark.parametrize("postorder", [False, True])
@pytest.mark.parametrize("batch_shape", [(), (4,)])
def test_native_gradient_matches_reference(postorder, batch_shape):
    problem = make_affine_problem(batch_shape=batch_shape, seed=17)
    ref_grads = _grads(problem, native=False, postorder=postorder)
    native_grads = _grads(problem, native=True, postorder=postorder)
    for ref_grad, native_grad in zip(ref_grads, native_grads):
        assert native_grad is not None
        assert native_grad.shape == ref_grad.shape
        assert_allclose(native_grad.numpy(), ref_grad.numpy(), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("postorder", [False, True])
def test_native_gradient_finite_difference(affine_problem, postorder):
    """Spot-check the analytic gradient against central finite differences."""
    problem = affine_problem
    loss_weights = tf.range(1, problem["node_count"] + 1, dtype=problem["x"].dtype)
    fn = _native_postorder if postorder else _native_preorder

    def loss_of(x):
        return tf.reduce_sum(fn(problem, x=x) * loss_weights)

    x_var = tf.Variable(problem["x"])
    with tf.GradientTape() as tape:
        loss = loss_of(x_var)
    analytic = tape.gradient(loss, x_var).numpy().reshape(-1)

    eps = 1e-6
    flat = problem["x"].numpy().reshape(-1).copy()
    for index in range(flat.size):
        plus, minus = flat.copy(), flat.copy()
        plus[index] += eps
        minus[index] -= eps
        finite_difference = (
            loss_of(tf.constant(plus.reshape(problem["x"].shape))).numpy()
            - loss_of(tf.constant(minus.reshape(problem["x"].shape))).numpy()
        ) / (2 * eps)
        assert_allclose(analytic[index], finite_difference, rtol=1e-5, atol=1e-7)


def test_native_root_parent_weight_gradient_is_zero(affine_problem):
    """The root's parent weight does not enter the forward map."""
    weight = tf.Variable(affine_problem["parent_weight"])
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(_native_preorder(affine_problem, weight=weight))
    assert tape.gradient(loss, weight).numpy()[-1] == 0.0


def test_native_broadcast_parameter_gradient_reduces():
    """A single (unbatched) parameter shared across a batch of coordinates: its
    gradient must reduce back to its own ``[node]`` shape."""
    problem = make_affine_problem(leaf_count=9, batch_shape=(6,), seed=14)
    scale = tf.Variable(problem["scale"])
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(_native_preorder(problem, scale=scale))
    with tf.GradientTape() as reference_tape:
        reference_loss = tf.reduce_sum(_reference_preorder(problem, scale=scale))
    native_grad = tape.gradient(loss, scale)
    reference_grad = reference_tape.gradient(reference_loss, scale)
    assert tuple(native_grad.shape) == (problem["node_count"],)
    assert_allclose(native_grad.numpy(), reference_grad.numpy(), rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Integration through the bijectors
# ---------------------------------------------------------------------------


def _affine_bijectors(problem, use_native):
    from treeflow.bijectors.tree_affine_bijector import (
        PostorderAffineBijector,
        PreorderAffineBijector,
    )

    return (
        PreorderAffineBijector(
            problem["topology"],
            problem["scale"],
            problem["shift"],
            problem["parent_weight"],
            use_native=use_native,
        ),
        PostorderAffineBijector(
            problem["topology"],
            problem["scale"],
            problem["shift"],
            problem["child_weight"],
            use_native=use_native,
        ),
    )


def test_bijector_native_matches_default_forward(affine_problem):
    for native, default in zip(
        _affine_bijectors(affine_problem, use_native=True),
        _affine_bijectors(affine_problem, use_native=False),
    ):
        assert_allclose(
            native.forward(affine_problem["x"]).numpy(),
            default.forward(affine_problem["x"]).numpy(),
            rtol=1e-12,
            atol=1e-12,
        )


def test_bijector_auto_uses_native_when_available(affine_problem):
    # The native op is built for this package (see conftest), so auto resolves
    # to using it.
    for bijector in _affine_bijectors(affine_problem, use_native="auto"):
        assert bijector._use_native is True


def test_bijector_invalid_use_native(affine_problem):
    with pytest.raises(ValueError):
        _affine_bijectors(affine_problem, use_native="sometimes")


def test_bijector_native_roundtrip(affine_problem):
    """Inverse (pure TF gather) composed with the native forward is the identity."""
    for bijector in _affine_bijectors(affine_problem, use_native=True):
        y = bijector.forward(affine_problem["x"])
        assert_allclose(
            bijector.inverse(y).numpy(),
            affine_problem["x"].numpy(),
            rtol=1e-9,
            atol=1e-9,
        )


def test_tree_normalizing_flow_native_matches_default(affine_problem):
    """The whole flow gives the same value, density and gradient either way."""
    from treeflow.bijectors.tree_normalizing_flow import (
        TreeFlowParameters,
        TreeNormalizingFlowBijector,
    )

    rng = np.random.default_rng(23)
    flow_parameters = TreeFlowParameters(
        node_count=affine_problem["node_count"], num_layers=2, dtype=tf.float64
    )
    for variable in flow_parameters.trainable_variables:
        variable.assign(
            variable + tf.constant(rng.normal(scale=0.3, size=variable.shape))
        )

    def build(use_native):
        return TreeNormalizingFlowBijector(
            affine_problem["topology"],
            flow_parameters=flow_parameters,
            use_native=use_native,
        )

    x = tf.Variable(affine_problem["x"])
    values, gradients = [], []
    for use_native in (True, False):
        bijector = build(use_native)
        with tf.GradientTape() as tape:
            value = tf.reduce_sum(
                bijector.forward(x)
            ) + bijector.forward_log_det_jacobian(x, event_ndims=1)
        values.append(value.numpy())
        gradients.append(tape.gradient(value, x).numpy())
    assert_allclose(values[0], values[1], rtol=1e-11, atol=1e-11)
    assert_allclose(gradients[0], gradients[1], rtol=1e-9, atol=1e-9)
