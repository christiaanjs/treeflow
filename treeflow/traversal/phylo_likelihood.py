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


def _combine_child_partials(node_child_transition_probs, child_partials, use_matvec):
    """Combine a node's children into its partial likelihood.

    For each child this is the matrix-vector product of the child's transition
    matrix with its partials (sum over the child state), followed by the product
    over children. The two forms below are mathematically identical (and give
    identical gradients); they only differ in performance:

    use_matvec
        ``False`` (default): explicit broadcast-multiply + ``reduce_sum``. Slower
        on the forward pass but its gradient is cheap, so it is faster for
        ``value + gradient`` -- the usual inference case -- and is the recommended
        default whenever gradients are required.
        ``True``: ``tf.linalg.matvec``, a fused contraction that is markedly faster
        on the *forward* pass (up to ~2x on large trees) but whose gradient is more
        expensive, making ``value + gradient`` slightly *slower*. Use it only for
        forward-only likelihood evaluation (no backprop).
    """
    if use_matvec:
        return tf.reduce_prod(
            tf.linalg.matvec(node_child_transition_probs, child_partials), axis=0
        )
    parent_child_probs = node_child_transition_probs * tf.expand_dims(
        child_partials, -2
    )
    return tf.reduce_prod(tf.reduce_sum(parent_child_probs, axis=-1), axis=0)


@tf.function
def phylogenetic_likelihood(
    sequences_onehot: tf.Tensor,
    transition_probs: tf.Tensor,
    frequencies: tf.Tensor,
    postorder_node_indices: tf.Tensor,
    child_indices: tf.Tensor,
    batch_shape=(),
    use_matvec: bool = False,
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
    use_matvec
        Combine each node's children with ``tf.linalg.matvec`` instead of the
        default broadcast-multiply + ``reduce_sum``. The result is identical.
        ``True`` is faster on the forward pass (up to ~2x on large trees) but has a
        more expensive gradient, so it is **slower for value + gradient** -- leave
        it ``False`` (the default) whenever gradients are needed; set it ``True``
        only for forward-only likelihood evaluation. See
        :func:`_combine_child_partials`.
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
        node_partials = _combine_child_partials(
            node_child_transition_probs, child_partials, use_matvec
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
    use_matvec: bool = False,
):
    """Numerically stable per-site phylogenetic LOG likelihood.

    Identical recursion to :func:`phylogenetic_likelihood`, but at every
    internal node the partial likelihood vector is divided by its per-site
    maximum and the log of that scale factor is accumulated. This prevents the
    partials from underflowing on large/deep trees. Returns the per-site ``log``
    likelihood rather than the linear likelihood.

    The scale factor is treated as a constant (``stop_gradient``): rescaling is
    an exact reparametrisation, so this yields the same gradient as the
    unrescaled likelihood while keeping the forward pass finite.

    Parameters match :func:`phylogenetic_likelihood`, including ``use_matvec``
    (faster forward, slower value+gradient -- leave ``False`` when gradients are
    needed).
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

    log_scale_sum = tf.zeros(batch_shape, dtype=transition_probs.dtype)
    for i in tf.range(taxon_count - 1):
        node_index = postorder_node_indices[i]
        node_child_transition_probs = child_transition_probs[i]
        node_child_indices = child_indices[i]
        child_partials = postorder_partials_ta.gather(node_child_indices)
        node_partials = _combine_child_partials(
            node_child_transition_probs, child_partials, use_matvec
        )
        # Rescale by the per-site maximum partial (treated as constant).
        scale = tf.reduce_max(node_partials, axis=-1, keepdims=True)
        scale = tf.where(scale > 0, scale, tf.ones_like(scale))
        scale = tf.stop_gradient(scale)
        node_partials = node_partials / scale
        log_scale_sum = log_scale_sum + tf.math.log(scale[..., 0])
        postorder_partials_ta = postorder_partials_ta.write(node_index, node_partials)

    root_partials = postorder_partials_ta.gather([2 * taxon_count - 2])[0]
    site_likelihood = tf.reduce_sum(frequencies * root_partials, axis=-1)
    return tf.math.log(site_likelihood) + log_scale_sum
