import tensorflow as tf
from treeflow.tree.topology.tensorflow_tree_topology import (
    TensorflowTreeTopology,
    numpy_topology_to_tensor,
)
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
from numpy.testing import assert_allclose


@tf.function
def function_tf(topology: TensorflowTreeTopology):
    return topology.child_indices[..., topology.taxon_count, :]


def function_np(topology: NumpyTreeTopology):
    return topology.child_indices[..., topology.taxon_count, :]


def test_TensorflowTopology_function_arg(flat_tree_test_data):
    numpy_topology = NumpyTreeTopology(
        parent_indices=flat_tree_test_data.parent_indices
    )
    tf_topology = numpy_topology_to_tensor(numpy_topology)
    res = function_tf(tf_topology)
    expected = function_np(numpy_topology)
    assert_allclose(res.numpy(), expected)


def test_TensorflowTopology_nest_map(flat_tree_test_data):
    numpy_topology = NumpyTreeTopology(
        parent_indices=flat_tree_test_data.parent_indices
    )
    tf_topology = numpy_topology_to_tensor(numpy_topology)
    res = tf.nest.map_structure(tf.shape, tf_topology)
