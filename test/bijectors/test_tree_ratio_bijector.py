from logging.handlers import DEFAULT_UDP_LOGGING_PORT
import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF
import numpy as np
from numpy.linalg import det as np_det
from numpy.testing import assert_allclose
from treeflow.tree.rooted.tensorflow_rooted_tree import (
    TensorflowRootedTree,
    convert_tree_to_tensor,
)
from treeflow.bijectors.tree_ratio_bijector import TreeRatioBijector
from treeflow_test_helpers.ratio_helpers import (
    topology_from_ratio_test_data,
    numpy_tree_from_ratio_test_data,
    RatioTestData,
)


def bijector_from_ratio_test_data(ratio_test_data: RatioTestData) -> TreeRatioBijector:
    return TreeRatioBijector(
        topology_from_ratio_test_data(ratio_test_data), ratio_test_data.anchor_heights
    )


def tree_from_ratio_test_data(ratio_test_data: RatioTestData) -> TensorflowRootedTree:
    return convert_tree_to_tensor(
        numpy_tree_from_ratio_test_data(ratio_test_data),
    )


def ratio_tree_from_ratio_test_data(
    ratio_test_data: RatioTestData,
) -> TensorflowRootedTree:
    return TensorflowRootedTree(
        topology=topology_from_ratio_test_data(ratio_test_data),
        node_heights=tf.convert_to_tensor(
            ratio_test_data.ratios, dtype=DEFAULT_FLOAT_DTYPE_TF
        ),
        sampling_times=ratio_test_data.sampling_times,
    )


# def test_TreeRatioBijector_inverse_dtype(flat_ratio_test_data: RatioTestData):
#     bijector = bijector_from_ratio_test_data(flat_ratio_test_data)
#     res = bijector.inverse_dtype()
#     assert isinstance(res, TensorflowRootedTree)
#     assert res.topology.parent_indices == tf.int32
#     assert res.topology.child_indices == tf.int32
#     assert res.topology.preorder_indices == tf.int32

#     assert res.node_heights == DEFAULT_FLOAT_DTYPE_TF
#     assert res.sampling_times == DEFAULT_FLOAT_DTYPE_TF


def logit(x):
    return -tf.math.log(1.0 / x - 1.0)


def unconstrain_node_height_ratios(node_heights):
    return tf.concat(
        [
            logit(node_heights[..., :-1]),
            tf.math.log(node_heights[..., -1:]),
        ],
        -1,
    )


def test_TreeRatioBijector_forward(ratio_test_data: RatioTestData):
    bijector = bijector_from_ratio_test_data(ratio_test_data)
    ratio_tree = ratio_tree_from_ratio_test_data(ratio_test_data)
    unconstrained_node_heights = unconstrain_node_height_ratios(ratio_tree.node_heights)
    unconstrained_tree = ratio_tree.with_node_heights(unconstrained_node_heights)
    res = bijector.forward(unconstrained_tree)
    assert_allclose(res.node_heights, ratio_test_data.heights)
    assert_allclose(res.topology.parent_indices, ratio_test_data.parent_indices)


def test_TreeRatioBijector_inverse(ratio_test_data: RatioTestData):
    bijector = bijector_from_ratio_test_data(ratio_test_data)
    tree = tree_from_ratio_test_data(ratio_test_data)
    res = bijector.inverse(tree)
    unconstrained_node_heights = unconstrain_node_height_ratios(ratio_test_data.ratios)
    assert_allclose(res.node_heights, unconstrained_node_heights, atol=1e-15)
    assert_allclose(res.topology.parent_indices, ratio_test_data.parent_indices)


def test_TreeRatioBijector_inverse_log_det_jacobian(
    ratio_test_data: RatioTestData,
):
    bijector = bijector_from_ratio_test_data(ratio_test_data)
    tree = tree_from_ratio_test_data(ratio_test_data)
    with tf.GradientTape() as t:
        t.watch(tree.node_heights)
        ratios_res = bijector.inverse(tree)

    if len(ratio_test_data.heights.shape) > 1:
        jac = t.batch_jacobian(ratios_res.node_heights, tree.node_heights)
    else:
        jac = t.jacobian(ratios_res.node_heights, tree.node_heights)
    log_det = np.log(np_det(jac.numpy()))
    res = bijector.inverse_log_det_jacobian(tree)
    assert_allclose(res.numpy(), log_det)
