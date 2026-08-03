"""Correctness tests for the affine tree maps of the tree normalising flow.

The forward sweeps are checked against a direct NumPy transcription of the
recursions, the inverses against a round trip, and the closed-form
log-det-Jacobian against the determinant of an autodiff Jacobian.
"""
import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.traversal.tree_affine import (
    affine_log_det_jacobian,
    node_child_indices,
    node_parent_indices,
    postorder_affine,
    postorder_affine_inverse,
    preorder_affine,
    preorder_affine_inverse,
)
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
from treeflow.tree.topology.tensorflow_tree_topology import numpy_topology_to_tensor

UNROLL_MODES = ["unrolled", "tensorarray", "while_loop"]


def random_parent_indices(taxon_count: int, rng: np.random.Generator) -> np.ndarray:
    """Random bifurcating topology with the children<parent index convention."""
    node_count = 2 * taxon_count - 1
    parent = np.full(node_count, -1, dtype=np.int32)
    active = list(range(taxon_count))
    next_node = taxon_count
    while len(active) > 1:
        a = active.pop(rng.integers(len(active)))
        b = active.pop(rng.integers(len(active)))
        parent[a] = next_node
        parent[b] = next_node
        active.append(next_node)
        next_node += 1
    return parent[:-1]  # drop the root, matching treeflow's parent_indices layout


def make_problem(taxon_count=8, batch_shape=(), seed=0):
    rng = np.random.default_rng(seed)
    topology = numpy_topology_to_tensor(
        NumpyTreeTopology(parent_indices=random_parent_indices(taxon_count, rng))
    )
    node_count = taxon_count - 1
    dtype = DEFAULT_FLOAT_DTYPE_TF.as_numpy_dtype
    shape = tuple(batch_shape) + (node_count,)

    def constant(x):
        return tf.constant(np.asarray(x, dtype=dtype))

    return dict(
        topology=topology,
        taxon_count=taxon_count,
        node_count=node_count,
        x=constant(rng.normal(size=shape)),
        scale=constant(rng.uniform(0.5, 1.5, size=(node_count,))),
        shift=constant(rng.normal(size=(node_count,))),
        parent_weight=constant(rng.uniform(-0.9, 0.9, size=(node_count,))),
        child_weight=constant(rng.uniform(-0.4, 0.4, size=(node_count, 2))),
    )


@pytest.fixture
def problem():
    return make_problem(seed=1)


def numpy_preorder(problem):
    topology = problem["topology"]
    taxon_count = problem["taxon_count"]
    preorder = topology.preorder_node_indices.numpy() - taxon_count
    parent = topology.parent_indices.numpy()[taxon_count:] - taxon_count
    x = problem["x"].numpy()
    scale, shift = problem["scale"].numpy(), problem["shift"].numpy()
    weight = problem["parent_weight"].numpy()
    y = np.zeros_like(x)
    root = preorder[0]
    y[..., root] = scale[root] * x[..., root] + shift[root]
    for i in preorder[1:]:
        y[..., i] = scale[i] * x[..., i] + shift[i] + weight[i] * y[..., parent[i]]
    return y


def numpy_postorder(problem):
    topology = problem["topology"]
    taxon_count = problem["taxon_count"]
    postorder = topology.postorder_node_indices.numpy() - taxon_count
    children = topology.node_child_indices.numpy() - taxon_count
    x = problem["x"].numpy()
    scale, shift = problem["scale"].numpy(), problem["shift"].numpy()
    weight = problem["child_weight"].numpy()
    w = np.zeros_like(x)
    for i in postorder:
        value = scale[i] * x[..., i] + shift[i]
        for c, child in enumerate(children[i]):
            if child >= 0:
                value = value + weight[i, c] * w[..., child]
        w[..., i] = value
    return w


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unroll", UNROLL_MODES)
def test_preorder_affine_matches_recursion(problem, unroll):
    result = preorder_affine(
        problem["topology"],
        problem["x"],
        problem["scale"],
        problem["shift"],
        problem["parent_weight"],
        unroll=unroll,
    )
    assert_allclose(result.numpy(), numpy_preorder(problem), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("unroll", UNROLL_MODES)
def test_postorder_affine_matches_recursion(problem, unroll):
    result = postorder_affine(
        problem["topology"],
        problem["x"],
        problem["scale"],
        problem["shift"],
        problem["child_weight"],
        unroll=unroll,
    )
    assert_allclose(result.numpy(), numpy_postorder(problem), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3)])
def test_batched_forward(batch_shape):
    problem = make_problem(batch_shape=batch_shape, seed=7)
    preorder_result = preorder_affine(
        problem["topology"],
        problem["x"],
        problem["scale"],
        problem["shift"],
        problem["parent_weight"],
    )
    postorder_result = postorder_affine(
        problem["topology"],
        problem["x"],
        problem["scale"],
        problem["shift"],
        problem["child_weight"],
    )
    expected_shape = batch_shape + (problem["node_count"],)
    assert tuple(preorder_result.shape) == expected_shape
    assert tuple(postorder_result.shape) == expected_shape
    assert_allclose(preorder_result.numpy(), numpy_preorder(problem), rtol=1e-12)
    assert_allclose(postorder_result.numpy(), numpy_postorder(problem), rtol=1e-12)


def test_batched_parameters(problem):
    """Per-sample parameters (a batched scale/shift/weight) broadcast too."""
    node_count = problem["node_count"]
    sample_shape = (4,)
    tile = lambda x: tf.tile(tf.expand_dims(x, 0), [sample_shape[0], 1])
    result = preorder_affine(
        problem["topology"],
        problem["x"],
        tile(problem["scale"]),
        tile(problem["shift"]),
        tile(problem["parent_weight"]),
    )
    assert tuple(result.shape) == sample_shape + (node_count,)
    assert_allclose(
        result.numpy(), np.broadcast_to(numpy_preorder(problem), result.shape)
    )


def test_root_parent_weight_is_unused(problem):
    """The root has no parent, so its ``parent_weight`` entry cannot matter."""
    perturbed = tf.concat(
        [problem["parent_weight"][:-1], problem["parent_weight"][-1:] + 3.0], axis=0
    )
    args = (problem["topology"], problem["x"], problem["scale"], problem["shift"])
    assert_allclose(
        preorder_affine(*args, problem["parent_weight"]).numpy(),
        preorder_affine(*args, perturbed).numpy(),
    )


# ---------------------------------------------------------------------------
# Inverses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_shape", [(), (5,)])
def test_preorder_roundtrip(batch_shape):
    problem = make_problem(batch_shape=batch_shape, seed=11)
    args = (problem["scale"], problem["shift"], problem["parent_weight"])
    y = preorder_affine(problem["topology"], problem["x"], *args)
    x = preorder_affine_inverse(problem["topology"], y, *args)
    assert_allclose(x.numpy(), problem["x"].numpy(), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("batch_shape", [(), (5,)])
def test_postorder_roundtrip(batch_shape):
    problem = make_problem(batch_shape=batch_shape, seed=13)
    args = (problem["scale"], problem["shift"], problem["child_weight"])
    y = postorder_affine(problem["topology"], problem["x"], *args)
    x = postorder_affine_inverse(problem["topology"], y, *args)
    assert_allclose(x.numpy(), problem["x"].numpy(), rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Log-det-Jacobian and dependence structure
# ---------------------------------------------------------------------------


def _jacobian(fn, x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = fn(x)
    return tape.jacobian(y, x).numpy()


def test_log_det_jacobian_matches_determinant(problem):
    x = problem["x"]  # unbatched, so the Jacobian is a plain [node, node] matrix
    expected = affine_log_det_jacobian(problem["scale"]).numpy()
    for jacobian in (
        _jacobian(
            lambda x: preorder_affine(
                problem["topology"],
                x,
                problem["scale"],
                problem["shift"],
                problem["parent_weight"],
            ),
            x,
        ),
        _jacobian(
            lambda x: postorder_affine(
                problem["topology"],
                x,
                problem["scale"],
                problem["shift"],
                problem["child_weight"],
            ),
            x,
        ),
    ):
        sign, log_abs_det = np.linalg.slogdet(jacobian)
        assert sign > 0
        assert_allclose(log_abs_det, expected, rtol=1e-10)


def test_dependence_directions(problem):
    """The preorder map propagates root-to-tip and the postorder map tip-to-root:
    in internal-node space each Jacobian is triangular in its own order, and
    they are transposes of each other's sparsity pattern."""
    topology = problem["topology"]
    x = problem["x"]
    parent = node_parent_indices(topology).numpy()
    children = node_child_indices(topology).numpy()

    preorder_jacobian = _jacobian(
        lambda x: preorder_affine(
            topology, x, problem["scale"], problem["shift"], problem["parent_weight"]
        ),
        x,
    )
    postorder_jacobian = _jacobian(
        lambda x: postorder_affine(
            topology, x, problem["scale"], problem["shift"], problem["child_weight"]
        ),
        x,
    )
    # A node's own preorder output depends on its parent's input, never the
    # other way round (and vice versa for the postorder map).
    for i, p in enumerate(parent):
        assert abs(preorder_jacobian[i, p]) > 0
        assert postorder_jacobian[i, p] == 0
    for i, node_children in enumerate(children):
        for child in node_children:
            if child >= 0:
                assert abs(postorder_jacobian[i, child]) > 0
                assert preorder_jacobian[i, child] == 0
