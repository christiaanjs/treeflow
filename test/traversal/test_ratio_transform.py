from numpy.testing import assert_allclose
from treeflow.traversal.ratio_transform import ratios_to_node_heights
import tensorflow as tf


def test_ratios_to_node_heights(ratio_test_data):
    res = ratios_to_node_heights(
        tf.constant(ratio_test_data.preorder_node_indices, dtype=tf.int32),
        tf.constant(ratio_test_data.node_parent_indices, dtype=tf.int32),
        ratio_test_data.ratios,
        ratio_test_data.anchor_heights,
    )

    assert_allclose(res, ratio_test_data.heights)
