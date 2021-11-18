import pytest
import tensorflow as tf
from treeflow.tree.topology.tensorflow_tree_topology import (
    TensorflowTreeTopology,
    numpy_topology_to_tensor,
)
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
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
