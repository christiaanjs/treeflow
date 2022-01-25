from numpy.testing import assert_allclose
from treeflow_test_helpers.ratio_helpers import (
    RatioTestData,
    numpy_tree_from_ratio_test_data,
)
from treeflow.traversal.anchor_heights import (
    get_anchor_heights,
    get_anchor_heights_tensor,
)
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor


def test_get_anchor_heights(ratio_test_data: RatioTestData):
    tree = numpy_tree_from_ratio_test_data(ratio_test_data)
    res = get_anchor_heights(tree)
    assert_allclose(res, ratio_test_data.anchor_heights)


def test_get_anchor_heights_tensor(ratio_test_data: RatioTestData):
    tree = convert_tree_to_tensor(numpy_tree_from_ratio_test_data(ratio_test_data))
    res = get_anchor_heights_tensor(tree.topology, tree.sampling_times)
    assert_allclose(res.numpy(), ratio_test_data.anchor_heights)
