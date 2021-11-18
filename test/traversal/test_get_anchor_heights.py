from numpy.testing import assert_allclose
from treeflow_test_helpers.ratio_helpers import (
    RatioTestData,
    numpy_tree_from_ratio_test_data,
)
from treeflow.traversal.anchor_heights import get_anchor_heights


def test_get_anchor_heights(ratio_test_data: RatioTestData):
    tree = numpy_tree_from_ratio_test_data(ratio_test_data)
    res = get_anchor_heights(tree)
    assert_allclose(res, ratio_test_data.anchor_heights)
