from collections import namedtuple
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree

TreeTestData = namedtuple(
    "TreeTestData",
    ["parent_indices", "node_heights", "sampling_times", "branch_lengths"],
)

from treeflow.tree.rooted.tensorflow_rooted_tree import (
    TensorflowRootedTree,
    convert_tree_to_tensor,
)


def test_data_to_tensor_tree(tree_test_data: TreeTestData) -> TensorflowRootedTree:
    numpy_tree = NumpyRootedTree(
        node_heights=tree_test_data.node_heights,
        sampling_times=tree_test_data.sampling_times,
        parent_indices=tree_test_data.parent_indices,
    )
    tf_tree = convert_tree_to_tensor(numpy_tree)
    return tf_tree
