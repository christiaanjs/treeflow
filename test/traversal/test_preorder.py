import pytest
from numpy.testing import assert_allclose
import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow_test_helpers.ratio_helpers import (
    topology_from_ratio_test_data,
    RatioTestData,
)
from treeflow.traversal.preorder import preorder_traversal
from treeflow.traversal.ratio_transform import move_outside_axis_to_inside
from tensorflow_probability.python.internal import distribution_util


def c(x):
    return tf.constant(x, dtype=DEFAULT_FLOAT_DTYPE_TF)


def move_inside_axis_to_outside(x):
    return distribution_util.move_dimension(x, -1, 0)


def ratios_to_node_heights_traversal(topology, ratios, anchor_heights):
    input = (
        move_inside_axis_to_outside(ratios),
        move_inside_axis_to_outside(anchor_heights),
    )

    def mapping(parent_height, input):
        ratio, anchor_height = input
        return (parent_height - anchor_height) * ratio + anchor_height

    init = input[0][-1] + input[1][-1]

    traversal_res = preorder_traversal(topology, mapping, input, init)
    return move_outside_axis_to_inside(traversal_res)


@pytest.mark.parametrize("function_mode", [True, False])
def test_preorder_traversal_ratio_transform(
    ratio_test_data: RatioTestData, function_mode: bool
):
    topology = topology_from_ratio_test_data(ratio_test_data)
    ratios = c(ratio_test_data.ratios)
    anchor_heights = c(ratio_test_data.anchor_heights)

    if function_mode:
        func = tf.function(ratios_to_node_heights_traversal)
    else:
        func = ratios_to_node_heights_traversal
    res = func(topology, ratios, anchor_heights)
    assert_allclose(res.numpy(), ratio_test_data.heights)
