import pytest
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.tree.rooted.tensorflow_rooted_tree import (
    TensorflowRootedTree,
    convert_tree_to_tensor,
)
import tensorflow as tf
from numpy.testing import assert_allclose


def test_data_to_tensor_tree(tree_test_data):
    numpy_tree = NumpyRootedTree(
        tree_test_data.heights, parent_indices=tree_test_data.parent_indices
    )
    tf_tree = convert_tree_to_tensor(numpy_tree)
    return tf_tree


def test_TensorflowRootedTree_from_numpy(tree_test_data):
    tf_tree = test_data_to_tensor_tree(tree_test_data)
    assert_allclose(tf_tree.heights.numpy(), tree_test_data.heights)
    assert_allclose(
        tf_tree.topology.parent_indices.numpy(), tree_test_data.parent_indices
    )


@pytest.mark.parametrize("function_mode", [True, False])
def test_TensorflowRootedTree_get_branch_lengths(function_mode, tree_test_data):
    """Also tests composite tensor functionality"""
    tf_tree = test_data_to_tensor_tree(tree_test_data)

    def blen_func(tree: TensorflowRootedTree):
        return tree.branch_lengths

    if function_mode:
        blen_func = tf.function(blen_func)

    blen_result = blen_func(tf_tree)
    assert_allclose(tree_test_data.branch_lengths, blen_result.numpy())
