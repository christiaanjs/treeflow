import pytest
import tensorflow as tf
from treeflow.tree.topology.tensorflow_tree_topology import (
    TensorflowTreeTopology,
    compute_preorder_indices,
    numpy_topology_to_tensor,
)
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
import treeflow.tree.topology.numpy_topology_operations as np_top_ops
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose
import numpy as np


@tf.function
def function_tf(topology: TensorflowTreeTopology):
    return topology.child_indices[..., topology.taxon_count, :]


def function_np(topology: NumpyTreeTopology):
    return topology.child_indices[..., topology.taxon_count, :]


def test_TensorflowTreeTopology_function_arg(flat_tree_test_data):
    numpy_topology = NumpyTreeTopology(
        parent_indices=flat_tree_test_data.parent_indices
    )
    tf_topology = numpy_topology_to_tensor(numpy_topology)
    res = function_tf(tf_topology)
    expected = function_np(numpy_topology)
    assert_allclose(res.numpy(), expected)


def test_TensorflowTreeTopology_nest_map(flat_tree_test_data):
    numpy_topology = NumpyTreeTopology(
        parent_indices=flat_tree_test_data.parent_indices
    )
    tf_topology = numpy_topology_to_tensor(numpy_topology)
    res = tf.nest.map_structure(tf.shape, tf_topology)


def test_TensorflowTreeTopology_get_prefer_static_rank(flat_tree_test_data):
    numpy_topology = NumpyTreeTopology(
        parent_indices=flat_tree_test_data.parent_indices
    )
    tf_topology = numpy_topology_to_tensor(numpy_topology)
    rank = tf_topology.get_prefer_static_rank()
    assert isinstance(rank.parent_indices, np.ndarray)
    assert isinstance(rank.preorder_indices, np.ndarray)
    assert isinstance(rank.child_indices, np.ndarray)
    assert rank.parent_indices == 1
    assert rank.preorder_indices == 1
    assert rank.child_indices == 2


@pytest.mark.parametrize(
    ["rank", "expected"],
    [
        (
            TensorflowTreeTopology(
                parent_indices=1, preorder_indices=1, child_indices=2
            ),
            False,
        ),  # Static shape, no batch
        (
            TensorflowTreeTopology(
                parent_indices=1, preorder_indices=1, child_indices=4
            ),
            True,
        ),  # Static shape, with batch
    ],
)  # TODO: Tests for dynamic shape?
def test_TensorflowTreeTopology_has_rank_to_has_batch_dimensions_static(rank, expected):
    res = TensorflowTreeTopology.rank_to_has_batch_dimensions(rank)
    assert res == expected


# ── compute_preorder_indices ─────────────────────────────────────────────────


def test_compute_preorder_indices_unbatched_matches_numpy(flat_tree_test_data):
    """TF preorder matches np_top_ops output on a single topology."""
    np_child = np_top_ops.get_child_indices(flat_tree_test_data.parent_indices)
    expected = np_top_ops.get_preorder_indices(np_child)
    result = compute_preorder_indices(tf.constant(np_child, dtype=tf.int32))
    assert result.shape == expected.shape
    assert_array_equal(result.numpy(), expected)


def test_compute_preorder_indices_batched_matches_numpy():
    """TF batched preorder matches np_top_ops on a batch of random topologies."""
    from treeflow.distributions.tree.birthdeath.cpp_sampler import build_random_topologies

    seed = tf.constant([0, 1], dtype=tf.int32)
    topo = build_random_topologies(30, 5, seed)

    child_np = topo.child_indices.numpy()  # [30, 9, 2]
    expected = np_top_ops.get_preorder_indices(child_np)  # [30, 9]

    result = compute_preorder_indices(topo.child_indices)  # [30, 9]
    assert result.shape == expected.shape
    assert_array_equal(result.numpy(), expected)


def test_compute_preorder_indices_root_is_first(flat_tree_test_data):
    """Root (last node, index node_count-1) must appear first in preorder."""
    np_child = np_top_ops.get_child_indices(flat_tree_test_data.parent_indices)
    result = compute_preorder_indices(tf.constant(np_child, dtype=tf.int32))
    node_count = np_child.shape[0]
    assert result.numpy()[0] == node_count - 1


@pytest.mark.parametrize("n_taxa", [2, 4, 7])
def test_compute_preorder_indices_contains_all_nodes(n_taxa):
    """Every node index 0..node_count-1 appears exactly once in preorder."""
    from treeflow.distributions.tree.birthdeath.cpp_sampler import build_random_topologies

    seed = tf.constant([7, 42], dtype=tf.int32)
    topo = build_random_topologies(10, n_taxa, seed)
    node_count = 2 * n_taxa - 1
    result = compute_preorder_indices(topo.child_indices)  # [10, node_count]
    for s in range(10):
        row = sorted(result.numpy()[s].tolist())
        assert row == list(range(node_count)), (
            f"n_taxa={n_taxa}, sample {s}: preorder {row} != {list(range(node_count))}"
        )


def test_compute_preorder_indices_inside_tf_function(flat_tree_test_data):
    """compute_preorder_indices runs correctly inside tf.function."""
    np_child = np_top_ops.get_child_indices(flat_tree_test_data.parent_indices)
    child_tf = tf.constant(np_child, dtype=tf.int32)
    expected = np_top_ops.get_preorder_indices(np_child)

    @tf.function
    def fn(ci):
        return compute_preorder_indices(ci)

    result = fn(child_tf)
    assert_array_equal(result.numpy(), expected)


def test_numpy_topology_to_tensor_preorder_matches_numpy(flat_tree_test_data):
    """numpy_topology_to_tensor now uses compute_preorder_indices; verify output."""
    numpy_topology = NumpyTreeTopology(
        parent_indices=flat_tree_test_data.parent_indices
    )
    np_child = np_top_ops.get_child_indices(flat_tree_test_data.parent_indices)
    expected_preorder = np_top_ops.get_preorder_indices(np_child)

    tf_topology = numpy_topology_to_tensor(numpy_topology)
    assert_array_equal(tf_topology.preorder_indices.numpy(), expected_preorder)
