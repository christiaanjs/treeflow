from numpy.testing import assert_allclose
from treeflow.traversal.ratio_transform import ratios_to_node_heights


def test_ratios_to_node_heights(ratio_test_data):
    res = ratios_to_node_heights(
        ratio_test_data.preorder_node_indices,
        ratio_test_data.node_parent_indices,
        ratio_test_data.ratios,
        ratio_test_data.anchor_heights,
    )

    assert_allclose(res, ratio_test_data.heights)
