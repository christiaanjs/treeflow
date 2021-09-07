import numpy as np
import pytest
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from numpy.testing import assert_allclose

branch_lengths_flat = [
    np.array(x) for x in [[0.3, 0.4, 1.2, 0.7], [0.9, 0.2, 2.3, 1.4]]
]
heights_flat = [
    np.array(x) for x in [[0.2, 0.1, 0.0, 0.5, 1.2], [0.0, 0.7, 0.0, 0.9, 2.3]]
]
parent_indices_single = np.array([3, 3, 4, 4])
branch_lengths_stacked = np.stack(branch_lengths_flat)
heights_stacked = np.stack(heights_flat)
parent_indices_stacked = np.stack([parent_indices_single, parent_indices_single])

heights = heights_flat + ([heights_stacked] * 2)
branch_lengths = branch_lengths_flat + ([branch_lengths_stacked] * 2)
parent_indices = ([parent_indices_single] * 3) + [np.stack([parent_indices_single] * 2)]


@pytest.mark.parametrize(
    ["heights", "parent_indices", "expected_branch_lengths"],
    zip(heights, parent_indices, branch_lengths),
)
def test_numpy_tree_get_branch_lengths(
    heights, parent_indices, expected_branch_lengths
):
    topology = NumpyTreeTopology(parent_indices)
    tree = NumpyRootedTree(topology, heights)
    branch_lengths = tree.branch_lengths
    assert_allclose(branch_lengths, expected_branch_lengths)
