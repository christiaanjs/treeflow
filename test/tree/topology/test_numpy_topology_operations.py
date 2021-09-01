import numpy as np
import pytest
from treeflow.tree.topology.numpy_topology_operations import get_child_indices
from numpy.testing import assert_equal

flat_parent_indices = [np.array(x) for x in [[4, 4, 5, 5, 6, 6], [4, 4, 5, 6, 5, 6]]]
taxon_count = (flat_parent_indices[0].shape[-1] + 2) // 2
leaf_child_indices = [[-1, -1]] * taxon_count
flat_expected_child_indices = [
    np.array(x)
    for x in [
        leaf_child_indices + [[0, 1], [2, 3], [4, 5]],
        leaf_child_indices + [[0, 1], [2, 4], [3, 5]],
    ]
]

parent_indices = flat_parent_indices + [np.stack(flat_parent_indices)]
expected_child_indices = flat_expected_child_indices + [
    np.stack(flat_expected_child_indices)
]


@pytest.mark.parametrize(
    ["parent_indices", "expected_child_indices"],
    zip(parent_indices, expected_child_indices),
)
def test_get_child_indices(
    parent_indices: np.ndarray, expected_child_indices: np.ndarray
):
    child_indices = get_child_indices(parent_indices)
    assert_equal(child_indices, expected_child_indices)
