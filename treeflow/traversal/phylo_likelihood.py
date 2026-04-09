import tensorflow as tf
from tensorflow_probability.python.internal import prefer_static as ps


def move_indices_to_outside(x, start, size):
    rank = tf.shape(tf.shape(x))[0]
    indices = tf.range(rank)
    end = start + size
    return tf.transpose(
        x,
        tf.concat([indices[start:end], indices[:start], indices[end:]], 0),
    )


@tf.function
def phylogenetic_likelihood(
    sequences_onehot: tf.Tensor,
    transition_probs: tf.Tensor,
    frequencies: tf.Tensor,
    postorder_node_indices: tf.Tensor,
    child_indices: tf.Tensor,
    batch_shape=(),
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
    probs_shape = ps.shape(transition_probs)
    state_shape = probs_shape[-1:]
    partials_shape = ps.concat([batch_shape, state_shape], axis=0)
    batch_rank = tf.shape(batch_shape)[0]

    postorder_partials_ta = tf.TensorArray(
        dtype=transition_probs.dtype,
        size=(2 * taxon_count - 1),
        element_shape=None if isinstance(partials_shape, tf.Tensor) else partials_shape,
    )
    child_transition_probs = move_indices_to_outside(
        tf.gather(transition_probs, child_indices, axis=-3), batch_rank, 2
    )  # [node, child, ..., state, state]

    for i in tf.range(taxon_count):
        postorder_partials_ta = postorder_partials_ta.write(
            i, sequences_onehot[..., i, :]
        )

    for i in tf.range(taxon_count - 1):
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


@tf.function
def phylogenetic_log_likelihood_rescaled(
    sequences_onehot: tf.Tensor,
    transition_probs: tf.Tensor,
    frequencies: tf.Tensor,
    postorder_node_indices: tf.Tensor,
    child_indices: tf.Tensor,
    batch_shape=(),
):
    """
    Per-site phylogenetic log-likelihood using rescaling for numerical stability.

    At each internal node the partial likelihoods are divided by their maximum
    value and the log of that scale is accumulated.  The final result is:

        log(frequencies · rescaled_root_partials) + accumulated_log_scale

    This avoids underflow on large trees or with very small branch lengths.

    sequences_onehot
        Tensor with shape [..., leaf, state]
    transition_probs
        Tensor with shape [..., node, state, state]
        Must be broadcastable over batch dimensions.
    frequencies
        Tensor with shape [..., state]
    postorder_node_indices
        Tensor with shape [internal_node].
        Cannot have batch dimensions.
    child_indices
        Tensor with shape [internal_node, children]
    batch_shape
        Leading batch dimensions shared by all tensors.
    """
    taxon_count = tf.shape(sequences_onehot)[-2]
    probs_shape = ps.shape(transition_probs)
    state_shape = probs_shape[-1:]
    partials_shape = ps.concat([batch_shape, state_shape], axis=0)
    batch_rank = tf.shape(batch_shape)[0]

    postorder_partials_ta = tf.TensorArray(
        dtype=transition_probs.dtype,
        size=(2 * taxon_count - 1),
        element_shape=None if isinstance(partials_shape, tf.Tensor) else partials_shape,
    )
    child_transition_probs = move_indices_to_outside(
        tf.gather(transition_probs, child_indices, axis=-3), batch_rank, 2
    )  # [node, child, ..., state, state]

    for i in tf.range(taxon_count):
        postorder_partials_ta = postorder_partials_ta.write(
            i, sequences_onehot[..., i, :]
        )

    log_scale = tf.zeros(batch_shape, dtype=transition_probs.dtype)

    for i in tf.range(taxon_count - 1):
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
        # Rescale: divide by max over states to prevent underflow.
        # When scale is zero the subtree is impossible; the final log(freq·root)
        # will correctly produce -inf in that case.
        scale = tf.reduce_max(node_partials, axis=-1, keepdims=True)
        safe_scale = tf.where(scale > 0, scale, tf.ones_like(scale))
        node_partials_rescaled = node_partials / safe_scale
        log_scale = log_scale + tf.squeeze(tf.math.log(safe_scale), axis=-1)
        postorder_partials_ta = postorder_partials_ta.write(
            node_index, node_partials_rescaled
        )

    root_partials = postorder_partials_ta.gather([2 * taxon_count - 2])[0]
    return (
        tf.math.log(tf.reduce_sum(frequencies * root_partials, axis=-1)) + log_scale
    )
