import tensorflow as tf


def phylogenetic_likelihood(
    sequences_onehot: tf.Tensor,
    transition_probs: tf.Tensor,
    frequencies: tf.Tensor,
    postorder_node_indices: tf.Tensor,
    child_indices: tf.Tensor,
):
    """
    Per-site phylogenetic likelihood.
    Assumes all parameters are broadcastable w.r.t. batch shape

    sequences_onehot
        Tensor with shape [..., leaf, state]
    transition_probs
        Tensor with shape [...,  node, state, state]
        Must be broadcasted over sites
    frequencies
        Tensor with shape [..., state]
    postorder_node_indices
        Tensor with shape [internal_node].
        Cannot have batch dimensions.
    child_indices
        Tensor with shape [internal_node,  children]
    """
    taxon_count = tf.shape(sequences_onehot)[-2]
    probs_shape = tf.shape(transition_probs)
    batch_shape = probs_shape[:-3]
    state_shape = probs_shape[-2:]
    leaf_partials = tf.broadcast_to(
        tf.expand_dims(sequences_onehot, -2),
        tf.concat([tf.expand_dims(taxon_count, -1), batch_shape, state_shape]),
    )

    postorder_partials_ta = tf.TensorArray(
        dtype=transition_probs.dtype,
        size=(2 * taxon_count - 1),
        element_shape=tf.shape(leaf_partials)[1:],
    )
    child_transition_probs = tf.gather(transition_probs, child_indices, axis=-3)

    for i in tf.range(taxon_count):
        node_index = postorder_node_indices[i]
        node_child_transition_probs = child_transition_probs[
            i
        ]  # child, ..., parent char, child char
        node_child_indices = child_indices[i]
        child_partials = postorder_partials_ta.gather(
            node_child_indices
        )  # child, ..., child char
        parent_child_probs = node_child_transition_probs * tf.expand_dims(
            child_partials, -2
        )
        node_partials = tf.reduce_prod(
            tf.reduce_sum(
                parent_child_probs,
                axis=-1,
            ),
            axis=0,
        )
        postorder_partials_ta = postorder_partials_ta.write(node_index, node_partials)
    root_partials = postorder_partials_ta.gather([2 * taxon_count - 2])[0]
    return tf.reduce_sum(frequencies * root_partials, axis=-1)
