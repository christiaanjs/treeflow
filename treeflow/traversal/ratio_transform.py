import tensorflow as tf
from tensorflow_probability.python.internal import prefer_static as ps


def ratios_to_node_heights(
    preorder_node_indices: tf.Tensor,
    parent_indices: tf.Tensor,
    proportions: tf.Tensor,
    anchor_heights: tf.Tensor,
):
    node_count = proportions.shape[-1]
    node_heights_ta = tf.TensorArray(
        dtype=proportions.dtype,
        size=node_count,
        element_shape=proportions.shape[:-1],
    )

    node_heights_ta.write(node_count - 1, proportions[..., node_count - 1])
    for i in preorder_node_indices[1:]:
        parent_height = node_heights_ta.gather(
            tf.expand_dims(parent_indices[..., i], 0)
        )[..., 0]
        anchor_height = anchor_heights[..., i]
        proportion = proportions[..., i]
        node_height = (parent_height - anchor_height) * proportion + anchor_height
        node_heights_ta.write(i, node_height)

    return node_heights_ta.stack()
