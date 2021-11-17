from _pytest.config import parse_warning_filter
import pytest
import numpy as np
from collections import namedtuple
from treeflow import DEFAULT_FLOAT_DTYPE_NP

RatioTestData = namedtuple(
    "RatioTestData",
    [
        "heights",
        "node_parent_indices",
        "preorder_node_indices",
        "ratios",
        "anchor_heights",
    ],
)


# parent indices: [5, 5, 6, 6, 8, 7, 7, 8]
# sampling times: [0.1, 0.2, 0.0, 0.3, 0.2]
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
node_parent_indices = np.array([7, 7, 8]) - 5
preorder_node_indices = np.array([8, 7, 5, 6]) - 5


@pytest.fixture(
    params=[
        RatioTestData(
            heights=heights_element,
            node_parent_indices=node_parent_indices,
            preorder_node_indices=preorder_node_indices,
            ratios=ratios_element,
            anchor_heights=anchor_heights_element,
        )
        for heights_element, ratios_element, anchor_heights_element in zip(
            heights, ratios, anchor_heights
        )
    ]
)
def ratio_test_data(request):
    return request.param
