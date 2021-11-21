import numpy as np
from numpy.linalg import det as np_det
import tensorflow as tf
from treeflow.bijectors.node_height_ratio_bijector import NodeHeightRatioBijector
from numpy.testing import assert_allclose
from treeflow_test_helpers.ratio_helpers import (
    topology_from_ratio_test_data,
    RatioTestData,
)


def bijector_from_ratio_test_data(ratio_test_data: RatioTestData):
    return NodeHeightRatioBijector(
        topology_from_ratio_test_data(ratio_test_data), ratio_test_data.anchor_heights
    )


def test_NodeHeightRatioBijector_forward(ratio_test_data: RatioTestData):
    bijector = bijector_from_ratio_test_data(ratio_test_data)
    res = bijector.forward(ratio_test_data.ratios)
    assert_allclose(res, ratio_test_data.heights)


def test_NodeHeightRatioBijector_inverse(ratio_test_data: RatioTestData):
    bijector = bijector_from_ratio_test_data(ratio_test_data)
    res = bijector.inverse(ratio_test_data.heights)
    assert_allclose(res, ratio_test_data.ratios)


def test_NodeHeightRatioBijector_inverse_log_det_jacobian(
    ratio_test_data: RatioTestData,
):
    bijector = bijector_from_ratio_test_data(ratio_test_data)
    heights = tf.convert_to_tensor(ratio_test_data.heights)
    with tf.GradientTape() as t:
        t.watch(heights)
        ratios_res = bijector.inverse(heights)

    if len(ratio_test_data.heights.shape) > 1:
        jac = t.batch_jacobian(ratios_res, heights)
    else:
        jac = t.jacobian(ratios_res, heights)
    log_det = np.log(np_det(jac.numpy()))
    res = bijector.inverse_log_det_jacobian(heights)
    assert_allclose(res.numpy(), log_det)