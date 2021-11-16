import pytest
from collections import namedtuple
from treeflow import DEFAULT_FLOAT_DTYPE_NP

RatioTestData = namedtuple(
    "RatioTestData", ["heights", "parent_indices", "preorder_indices", "ratios"]
)


# parent indices: [5, 5, 6, 6, 8, 7, 7, 8]
# sampling times: [0.1, 0.2, 0.0, 0.3, 0.2]
anchor_heights = [[0.0, 0.0, 0.0, 0.0], [0.2, 0.3, 0.3, 0.3]]
heights = [[0.2, 0.5, 0.8, 1.6], [0.6, 0.75, 1.2, 1.5]]
ratios = [
    [0.25, 0.625, 0.5, 1.6],
    [0.4, 0.5, 0.75, 1.2],
]
node_parent_indices = [7, 7, 8]
preorder_node_indices = [8, 7, 5, 6]


@pytest.fixture(params=[])
def ratio_test_data(request):
    return request.param
