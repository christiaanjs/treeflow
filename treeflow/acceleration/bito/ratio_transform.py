import numpy as np
import tensorflow as tf
import bito
from treeflow import DEFAULT_FLOAT_DTYPE_TF


def ratios_to_node_heights_numpy(inst, x):
    tree = inst.tree_collection.trees[0]
    node_height_state = np.array(tree.node_heights, copy=False)
    tree.initialize_time_tree_using_height_ratios(x)
    return node_height_state[-x.shape[-1] :].astype(x.dtype)


def ratio_gradient_numpy(inst, heights, dheights):
    tree = inst.tree_collection.trees[0]
    node_height_state = np.array(tree.node_heights, copy=False)
    node_height_state[-heights.shape[-1] :] = heights
    return np.array(
        bito.ratio_gradient_of_height_gradient(tree, dheights),
        dtype=heights.dtype,
    )


def ratios_to_node_heights(inst, anchor_heights, ratios):
    def numpy_func(ratios):
        return ratios_to_node_heights_numpy(inst, ratios)
    def numpy_grad_func(heights, dheights):
        return ratio_gradient_numpy(inst, heights, dheights)
    @tf.custom_gradient
    def bito_tf_func(x):
        heights = tf.numpy_function(
            numpy_func,
            [x],
            DEFAULT_FLOAT_DTYPE_TF,
        )

        def grad(dheights):
            return tf.numpy_function(
                numpy_grad_func,
                [heights, dheights],
                DEFAULT_FLOAT_DTYPE_TF,
            )

        return heights, grad

    with_root_height = tf.concat(
        [ratios[:-1], ratios[-1:] + anchor_heights[-1]], axis=0
    )

    # Libsbn doesn't add root bound
    return bito_tf_func(with_root_height)
