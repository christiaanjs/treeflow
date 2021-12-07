import pytest
import numpy as np
from treeflow_test_helpers.ratio_helpers import RatioTestData

sampling_times_flat = [
    np.array(x) for x in [[0.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.0, 0.3, 0.2]]
]
sampling_times = sampling_times_flat + [np.stack(sampling_times_flat)]
anchor_heights_flat = [
    np.array(x) for x in [[0.0, 0.0, 0.0, 0.0], [0.2, 0.3, 0.3, 0.3]]
]
anchor_heights = anchor_heights_flat + [np.stack(anchor_heights_flat)]
heights_flat = [np.array(x) for x in [[0.2, 0.5, 0.8, 1.6], [0.6, 0.75, 1.2, 1.5]]]
heights = heights_flat + [np.stack(heights_flat)]
ratios_flat = [
    np.array(x)
    for x in [
        [0.25, 0.625, 0.5, 1.6],
        [0.4, 0.5, 0.75, 1.2],
    ]
]
ratios = ratios_flat + [np.stack(ratios_flat)]
parent_indices = np.array([5, 5, 6, 6, 8, 7, 7, 8])
taxon_count = 5
node_parent_indices = parent_indices[taxon_count:] - taxon_count
preorder_node_indices = np.array([8, 7, 5, 6]) - 5


@pytest.fixture(
    params=[
        RatioTestData(
            heights=heights_element,
            parent_indices=parent_indices,
            node_parent_indices=node_parent_indices,
            preorder_node_indices=preorder_node_indices,
            ratios=ratios_element,
            anchor_heights=anchor_heights_element,
            sampling_times=sampling_times_element,
        )
        for heights_element, ratios_element, anchor_heights_element, sampling_times_element in zip(
            heights, ratios, anchor_heights, sampling_times
        )
    ]
)
def ratio_test_data(request):
    return request.param


@pytest.fixture
def flat_ratio_test_data():
    return RatioTestData(
        heights=heights[0],
        parent_indices=parent_indices,
        node_parent_indices=node_parent_indices,
        preorder_node_indices=preorder_node_indices,
        ratios=ratios[0],
        anchor_heights=anchor_heights[0],
        sampling_times=sampling_times[0],
    )
