import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF
import numpy as np
from numpy.testing import assert_allclose

from tensorflow_probability.python.bijectors import Chain, Shift, Scale
from treeflow_test_helpers.ratio_helpers import (
    topology_from_ratio_test_data,
    RatioTestData,
)
from treeflow.bijectors.preorder_node_bijector import PreorderNodeBijector
from treeflow.bijectors.node_height_ratio_bijector import NodeHeightRatioBijector


def ratio_transform_forward_mapping(parent_height, anchor_height):
    return Chain([Shift(anchor_height), Scale(parent_height - anchor_height)])


def ratio_transform_forward_root_mapping(anchor_height):
    return Shift(anchor_height)


def get_bijector(ratio_test_data: RatioTestData) -> PreorderNodeBijector:
    topology = topology_from_ratio_test_data(ratio_test_data)
    bijector = PreorderNodeBijector(
        topology,
        ratio_test_data.anchor_heights,
        ratio_transform_forward_mapping,
        ratio_transform_forward_root_mapping,
    )
    return bijector


def test_preorder_node_bijector_forward(ratio_test_data: RatioTestData):
    bijector = get_bijector(ratio_test_data)
    res = bijector.forward(ratio_test_data.ratios)
    expected = ratio_test_data.heights
    assert_allclose(res.numpy(), expected)


def test_preorder_node_bijector_forward_log_det_jacobian(
    ratio_test_data: RatioTestData,
):
    bijector = get_bijector(ratio_test_data)
    res = bijector.forward_log_det_jacobian(ratio_test_data.ratios)

    test_bijector = NodeHeightRatioBijector(
        bijector._topology, ratio_test_data.anchor_heights
    )
    expected = test_bijector.forward_log_det_jacobian(ratio_test_data.ratios)

    assert_allclose(res.numpy(), expected.numpy())


def test_preorder_node_bijector_inverse(ratio_test_data: RatioTestData):
    bijector = get_bijector(ratio_test_data)
    res = bijector.inverse(tf.constant(ratio_test_data.heights, DEFAULT_FLOAT_DTYPE_TF))
    assert_allclose(res.numpy(), ratio_test_data.ratios)
