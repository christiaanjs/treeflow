import treeflow.tree_processing
import treeflow.tree_transform
import treeflow.libsbn
import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose
from treeflow import DEFAULT_FLOAT_DTYPE_TF

def get_transforms(newick_file):
    tree_info = treeflow.tree_processing.parse_tree_info(newick_file)
    transform_args = treeflow.tree_transform.get_transform_args(tree_info)
    tf_transform = treeflow.tree_transform.BranchBreaking(**transform_args)
    
    inst = treeflow.libsbn.get_instance(newick_file)
    libsbn_transform = treeflow.tree_transform.Ratio(inst, **transform_args)
    return tf_transform, libsbn_transform, tree_info

def get_ratios(taxon_count):
    return (np.arange(taxon_count - 1) + 1.0) / taxon_count

def test_libsbn_transform_forward(newick_file):
    tf_transform, libsbn_transform, tree_info = get_transforms(newick_file)
    ratios = get_ratios(tree_info.node_bounds.shape[0] + 1)
    tf_heights = tf_transform.forward(ratios)
    libsbn_heights = libsbn_transform.forward(ratios)
    assert_allclose(tf_heights, libsbn_heights)

def get_test_gradient(ratios, transform):
    with tf.GradientTape() as t:
        t.watch(ratios)
        heights = transform.forward(ratios)
        res = tf.reduce_sum(heights)# ** 2)
    return t.gradient(res, ratios)

def test_libsbn_transform_gradient(newick_file):
    tf_transform, libsbn_transform, tree_info = get_transforms(newick_file)
    ratios = tf.convert_to_tensor(get_ratios(tree_info.node_bounds.shape[0] + 1), dtype=DEFAULT_FLOAT_DTYPE_TF)
    tf_gradient = get_test_gradient(ratios, tf_transform)
    libsbn_gradient = get_test_gradient(ratios, libsbn_transform)
    assert_allclose(tf_gradient, libsbn_gradient)
