import pytest
import numpy as np
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.tree.rooted.tensorflow_rooted_tree import (
    TensorflowRootedTree,
    convert_tree_to_tensor,
)
import tensorflow as tf
from numpy.testing import assert_allclose
from treeflow_test_helpers.tree_helpers import TreeTestData, data_to_tensor_tree


def test_TensorflowRootedTree_from_numpy(tree_test_data: TreeTestData):
    tf_tree = data_to_tensor_tree(tree_test_data)
    assert_allclose(tf_tree.sampling_times.numpy(), tree_test_data.sampling_times)
    assert_allclose(tf_tree.node_heights.numpy(), tree_test_data.node_heights)
    assert_allclose(
        tf_tree.topology.parent_indices.numpy(), tree_test_data.parent_indices
    )


def test_TensorflowRootedTree_heights(tree_test_data: TreeTestData):
    tf_tree = data_to_tensor_tree(tree_test_data)
    heights_res = tf_tree.heights
    expected_heights = np.concatenate(
        (tree_test_data.sampling_times, tree_test_data.node_heights), axis=-1
    )
    assert_allclose(heights_res.numpy(), expected_heights)


@pytest.mark.parametrize("function_mode", [True, False])
def test_TensorflowRootedTree_get_branch_lengths(
    function_mode, tree_test_data: TreeTestData
):
    """Also tests composite tensor functionality"""
    tf_tree = data_to_tensor_tree(tree_test_data)

    def blen_func(tree: TensorflowRootedTree):
        return tree.branch_lengths

    if function_mode:
        blen_func = tf.function(blen_func)

    blen_result = blen_func(tf_tree)
    assert_allclose(tree_test_data.branch_lengths, blen_result.numpy())
