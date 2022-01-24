import pytest
import numpy as np
from treeflow_test_helpers.tree_helpers import TreeTestData


taxon_count = 3
branch_lengths_flat = [
    np.array(x) for x in [[0.3, 0.4, 1.2, 0.7], [0.9, 0.2, 2.3, 1.4]]
]
sampling_times_flat = [
    np.array(x)
    for x in [
        [
            0.2,
            0.1,
            0.0,
        ],
        [
            0.0,
            0.7,
            0.0,
        ],
    ]
]
node_heights_flat = [np.array(x) for x in [[0.5, 1.2], [0.9, 2.3]]]
parent_indices_single = np.array([3, 3, 4, 4])
branch_lengths_stacked = np.stack(branch_lengths_flat)
node_heights_stacked = np.stack(node_heights_flat)
sampling_times_stacked = np.stack(sampling_times_flat)
parent_indices_stacked = np.stack([parent_indices_single, parent_indices_single])

node_heights = node_heights_flat + ([node_heights_stacked] * 2)
sampling_times = sampling_times_flat + ([sampling_times_stacked] * 2)
branch_lengths = branch_lengths_flat + ([branch_lengths_stacked] * 2)
parent_indices = ([parent_indices_single] * 3) + [np.stack([parent_indices_single] * 2)]


@pytest.fixture(
    params=[
        TreeTestData(*args)
        for args in zip(parent_indices, node_heights, sampling_times, branch_lengths)
    ]
)
def tree_test_data(request):
    return request.param


_flat_tree_test_data = TreeTestData(
    parent_indices[0], node_heights[0], sampling_times[0], branch_lengths[0]
)


@pytest.fixture()
def flat_tree_test_data():
    return _flat_tree_test_data
