import pytest
import numpy as np
from collections import namedtuple

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

TreeTestData = namedtuple(
    "TreeTestData", ["parent_indices", "heights", "branch_lengths"]
)


@pytest.fixture(
    params=[
        TreeTestData(*args) for args in zip(parent_indices, heights, branch_lengths)
    ]
)
def tree_test_data(request):
    return request.param


_flat_tree_test_data = TreeTestData(parent_indices[0], heights[0], branch_lengths[0])


@pytest.fixture()
def flat_tree_test_data():
    return _flat_tree_test_data
