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
from treeflow.tree.topology.numpy_tree_topology import StaticNumpyTreeTopology

def c(x):
    return tf.constant(x, dtype=DEFAULT_FLOAT_DTYPE_TF)


def move_inside_axis_to_outside(x):
    return distribution_util.move_dimension(x, -1, 0)


def ratios_to_node_heights_traversal(
    topology, ratios, anchor_heights, unroll="auto", custom_gradient=False
):
    input = (
        move_inside_axis_to_outside(ratios),
        move_inside_axis_to_outside(anchor_heights),
    )

    def mapping(parent_height, input):
        ratio, anchor_height = input
        return (parent_height - anchor_height) * ratio + anchor_height

    init = input[0][-1] + input[1][-1]

    traversal_res = preorder_traversal(
        topology, mapping, input, init, unroll=unroll, custom_gradient=custom_gradient
    )
    return move_outside_axis_to_inside(traversal_res)


@pytest.mark.parametrize("unroll", ["unrolled", "tensorarray", "while_loop"])
@pytest.mark.parametrize("function_mode", [True, False, "jit_compile"])
def test_preorder_traversal_ratio_transform(
    ratio_test_data: RatioTestData, function_mode: bool, unroll: str
):
    topology = topology_from_ratio_test_data(ratio_test_data)
    ratios = c(ratio_test_data.ratios)
    anchor_heights = c(ratio_test_data.anchor_heights)

    if unroll == "unrolled" and function_mode:
        topology = StaticNumpyTreeTopology.from_numpy_topology(topology.numpy())

    def gradient_func(topology, ratios, anchor_heights, **kwargs):
        with tf.GradientTape() as tape:
            tape.watch(ratios)
            res = tf.reduce_sum(
                ratios_to_node_heights_traversal(
                    topology, ratios, anchor_heights, **kwargs
                )
            )
        return tape.gradient(res, [ratios])

    if function_mode:
        func = tf.function(
            ratios_to_node_heights_traversal, jit_compile=function_mode == "jit_compile"
        )
        gradient_func = tf.function(
            gradient_func, jit_compile=function_mode == "jit_compile"
        )
    else:
        func = ratios_to_node_heights_traversal
    res = func(topology, ratios, anchor_heights, unroll=unroll)
    assert_allclose(res.numpy(), ratio_test_data.heights)

    grad_res = gradient_func(topology, ratios, anchor_heights, unroll=unroll)
    assert all(g is not None for g in grad_res)


@pytest.mark.parametrize("unroll", ["unrolled", "tensorarray", "while_loop"])
def test_preorder_custom_gradient_matches_autodiff(
    ratio_test_data: RatioTestData, unroll: str
):
    """The complementary-traversal custom gradient must match autodiff w.r.t. the
    traversal `input` (ratios and anchor heights)."""
    topology = topology_from_ratio_test_data(ratio_test_data)
    if unroll == "unrolled":
        topology = StaticNumpyTreeTopology.from_numpy_topology(topology.numpy())
    ratios = c(ratio_test_data.ratios)
    anchor_heights = c(ratio_test_data.anchor_heights)

    def value_and_grad(custom_gradient):
        r = tf.Variable(ratios)
        a = tf.Variable(anchor_heights)
        with tf.GradientTape() as tape:
            res = tf.reduce_sum(
                ratios_to_node_heights_traversal(
                    topology, r, a, unroll=unroll, custom_gradient=custom_gradient
                )
            )
        return res, tape.gradient(res, [r, a])

    res_auto, grad_auto = value_and_grad(False)
    res_custom, grad_custom = value_and_grad(True)

    assert_allclose(res_custom.numpy(), res_auto.numpy(), rtol=1e-12)
    assert all(g is not None for g in grad_custom)
    for g_custom, g_auto in zip(grad_custom, grad_auto):
        assert_allclose(g_custom.numpy(), g_auto.numpy(), rtol=1e-9, atol=1e-12)
