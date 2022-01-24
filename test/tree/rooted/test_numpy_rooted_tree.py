import numpy as np
import pytest
from treeflow_test_helpers.tree_helpers import TreeTestData
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from numpy.testing import assert_allclose


def test_numpy_tree_get_branch_lengths(tree_test_data: TreeTestData):
    expected_branch_lengths = tree_test_data.branch_lengths
    tree = NumpyRootedTree(
        sampling_times=tree_test_data.sampling_times,
        node_heights=tree_test_data.node_heights,
        parent_indices=tree_test_data.parent_indices,
    )
    branch_lengths = tree.branch_lengths
    assert_allclose(branch_lengths, expected_branch_lengths)
