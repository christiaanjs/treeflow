import tensorflow as tf
from tensorflow_probability.python.internal import prefer_static as ps


def move_outside_axis_to_inside(x):
    rank = tf.rank(x)
    perm = tf.concat([tf.range(1, rank), [0]], axis=0)
    return tf.transpose(x, perm)


@tf.function
def ratios_to_node_heights(
    preorder_node_indices: tf.Tensor,
    parent_indices: tf.Tensor,
    ratios: tf.Tensor,
    anchor_heights: tf.Tensor,
):
    # Written as an explicit bounded while_loop (static `maximum_iterations`) rather
    # than an AutoGraph `for i in <tensor>` loop so that the whole value-and-gradient
    # is XLA-compilable: XLA needs the TensorArray list size fixed, which the
    # unbounded AutoGraph form does not provide. This transform is only a few flops
    # per node, so XLA fusion is a large win (see the traversal-backends benchmark) --
    # but get it by `jit_compile`-ing the *whole* value+gradient at the call site
    # (e.g. the training/bijector step). Do NOT put `jit_compile=True` on this
    # function: callers differentiate through it with the tape outside, and the
    # forward's TensorArray (TensorList) cannot cross the XLA/TF boundary back into a
    # non-compiled backward ("TensorList crossing the XLA/TF boundary is not
    # implemented"). No read+stack is needed here: each step reads a single parent
    # (no TensorArray.gather), so the gather gradient that blocks XLA in the
    # postorder driver never arises.
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

    def cond(k, ta):
        return k < node_count

    def body(k, ta):
        i = preorder_node_indices[k]
        parent_height = ta.read(parent_indices[i])
        node_height = (
            parent_height - anchor_heights[..., i]
        ) * ratios[..., i] + anchor_heights[..., i]
        return k + 1, ta.write(i, node_height)

    _, node_heights_ta = tf.while_loop(
        cond,
        body,
        (tf.constant(1), node_heights_ta),
        maximum_iterations=node_count - 1,
    )

    return move_outside_axis_to_inside(node_heights_ta.stack())
