import tensorflow as tf
from tensorflow_probability.python.internal import prefer_static as ps


def move_outside_axis_to_inside(x):
    rank = tf.rank(x)
    axes = tf.range(rank)
    perm = tf.concat([tf.range(1, rank), [0]], axis=0)
    return tf.transpose(x, perm)


@tf.function
def ratios_to_node_heights(
    preorder_node_indices: tf.Tensor,
    parent_indices: tf.Tensor,
    ratios: tf.Tensor,
    anchor_heights: tf.Tensor,
):
    node_count = ratios.shape[-1]
    node_heights_ta = tf.TensorArray(
        dtype=ratios.dtype,
        size=node_count,
        element_shape=ratios.shape[:-1],
        clear_after_read=False,
    )

    node_heights_ta = node_heights_ta.write(
        node_count - 1,
        ratios[..., node_count - 1] + anchor_heights[..., node_count - 1],
    )
    for i in preorder_node_indices[1:]:
        parent_height = node_heights_ta.read(parent_indices[i])
        anchor_height = anchor_heights[..., i]
        proportion = ratios[..., i]
        node_height = (parent_height - anchor_height) * proportion + anchor_height
        node_heights_ta = node_heights_ta.write(i, node_height)

    return move_outside_axis_to_inside(node_heights_ta.stack())
