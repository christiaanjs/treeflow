import numpy as np
import tensorflow as tf
import pytest
from treeflow.bijectors.node_height_ratio_bijector import NodeHeightRatioBijector
from treeflow.tree.io import parse_newick
from treeflow.traversal.anchor_heights import get_anchor_heights
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from functools import partial
from numpy.testing import assert_allclose


def get_bito_forward_func(newick_file, dated):
    from treeflow.acceleration.bito.instance import get_instance, get_tree_info
    from treeflow.acceleration.bito.ratio_transform import (
        ratios_to_node_heights as bito_ratios_to_node_heights,
    )

    inst = get_instance(newick_file, dated=dated)
    tree, anchor_heights = get_tree_info(inst)
    return partial(
        bito_ratios_to_node_heights, inst=inst, anchor_heights=anchor_heights
    )


def get_treeflow_forward_func(numpy_tree, tensor_constant):
    anchor_heights = tensor_constant(get_anchor_heights(numpy_tree))
    return NodeHeightRatioBijector(
        convert_tree_to_tensor(numpy_tree).topology, anchor_heights
    ).forward


def get_ratios(taxon_count):
    return (np.arange(taxon_count - 1) + 1.0) / taxon_count


def get_test_values(forward_func, ratios):
    with tf.GradientTape() as t:
        t.watch(ratios)
        heights = forward_func(ratios)
        res = tf.reduce_sum(heights ** 2)

    grad = tf.gradients(res, ratios)
    return heights, grad


@pytest.mark.skip
def test_bito_ratio_transform_forward(newick_file_dated, tensor_constant):
    newick_file, dated = newick_file_dated
    numpy_tree = parse_newick(newick_file)
    bito_forward_func = get_bito_forward_func(newick_file, dated)
    treeflow_forward_func = get_treeflow_forward_func(numpy_tree, tensor_constant)
    ratios = tensor_constant(get_ratios(numpy_tree.taxon_count))

    treeflow_heights, treeflow_grad = get_test_values(treeflow_forward_func, ratios)
    bito_heights, bito_grad = get_test_values(bito_forward_func, ratios)

    assert_allclose(bito_heights.numpy(), treeflow_heights.numpy())
    assert_allclose(bito_grad.numpy(), treeflow_grad.numpy())
