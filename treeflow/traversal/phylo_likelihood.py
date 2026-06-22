import typing as tp
import tensorflow as tf
from treeflow.traversal.postorder import postorder_node_traversal
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology


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
    over children. The two forms are mathematically identical (same gradients);
    they only differ in performance.

    use_matvec
        ``False`` (default): explicit broadcast-multiply + ``reduce_sum`` -- cheaper
        gradient, so faster for value+gradient (the usual inference case).
        ``True``: ``tf.linalg.matvec`` -- faster forward pass but a more expensive
        gradient. Use only for forward-only evaluation.
    """
    if use_matvec:
        return tf.reduce_prod(
            tf.linalg.matvec(node_child_transition_probs, child_partials), axis=0
        )
    parent_child_probs = node_child_transition_probs * tf.expand_dims(
        child_partials, -2
    )
    return tf.reduce_prod(tf.reduce_sum(parent_child_probs, axis=-1), axis=0)


def _likelihood_partials(
    topology, sequences_onehot, transition_probs, batch_shape, mapping, leaf_extra,
    unroll,
):
    """Shared postorder traversal for the (un)rescaled likelihoods.

    ``leaf_extra`` is an optional extra leaf-init structure paired with the leaf
    partials (used by the rescaled variant to carry a per-node log-scale). Returns
    the stacked per-node output structure.
    """
    batch_rank = tf.shape(batch_shape)[0]
    # Per-internal-node child transition matrices: [n_internal, child, ..., state, state]
    child_transition_probs = move_indices_to_outside(
        tf.gather(transition_probs, topology.node_child_indices, axis=-3), batch_rank, 2
    )
    # Leaf partials with the leaf axis first: [leaf, ..., state]
    leaf_partials = move_indices_to_outside(
        sequences_onehot, tf.rank(sequences_onehot) - 2, 1
    )
    leaf_init = leaf_partials if leaf_extra is None else (leaf_partials, leaf_extra)
    return postorder_node_traversal(
        topology, mapping, child_transition_probs, leaf_init, unroll=unroll
    )


def phylogenetic_likelihood(
    topology: TensorflowTreeTopology,
    sequences_onehot: tf.Tensor,
    transition_probs: tf.Tensor,
    frequencies: tf.Tensor,
    batch_shape=(),
    use_matvec: bool = False,
    unroll: tp.Union[bool, str] = "auto",
):
    """Per-site phylogenetic likelihood, on the generic ``postorder_node_traversal``.

    Assumes all parameters are broadcastable w.r.t. batch shape.

    topology
        The tree topology (provides postorder/child indices; no batch dimensions).
    sequences_onehot
        Tensor with shape [..., leaf, state]
    transition_probs
        Tensor with shape [..., node, state, state]; broadcast over sites.
    frequencies
        Tensor with shape [..., state]
    use_matvec
        See :func:`_combine_child_partials`. ``True`` is forward-faster but slower for
        value+gradient; leave ``False`` when gradients are needed.
    unroll
        Forwarded to :func:`postorder_node_traversal`: ``"auto"`` unrolls when the
        topology is statically known, ``True`` forces it, ``False`` keeps the dynamic
        ``tf.while_loop``.
    """

    def mapping(child_output, node_input, topology_data):
        return _combine_child_partials(node_input, child_output, use_matvec)

    partials = _likelihood_partials(
        topology, sequences_onehot, transition_probs, batch_shape, mapping, None, unroll
    )
    return tf.reduce_sum(frequencies * partials[-1], axis=-1)


def phylogenetic_log_likelihood_rescaled(
    topology: TensorflowTreeTopology,
    sequences_onehot: tf.Tensor,
    transition_probs: tf.Tensor,
    frequencies: tf.Tensor,
    batch_shape=(),
    use_matvec: bool = False,
    unroll: tp.Union[bool, str] = "auto",
):
    """Numerically stable per-site phylogenetic LOG likelihood.

    Identical recursion to :func:`phylogenetic_likelihood`, but each internal node's
    partials are divided by their per-site maximum and the log of that scale factor is
    accumulated, preventing underflow on large/deep trees. The scale is treated as a
    constant (``stop_gradient``), so the gradient matches the unrescaled likelihood.

    The running scale is carried as a second component of the per-node output structure
    ``(partials, log_scale)`` (leaves contribute 0) and summed over nodes afterwards.
    Parameters match :func:`phylogenetic_likelihood`.
    """
    leaf_log_scale = tf.zeros(
        tf.concat([tf.shape(sequences_onehot)[-2:-1], batch_shape], 0),
        dtype=transition_probs.dtype,
    )

    def mapping(child_output, node_input, topology_data):
        child_partials, _child_log_scales = child_output
        node_partials = _combine_child_partials(node_input, child_partials, use_matvec)
        scale = tf.reduce_max(node_partials, axis=-1, keepdims=True)
        scale = tf.where(scale > 0, scale, tf.ones_like(scale))
        scale = tf.stop_gradient(scale)
        node_partials = node_partials / scale
        return node_partials, tf.math.log(scale[..., 0])

    partials, log_scales = _likelihood_partials(
        topology, sequences_onehot, transition_probs, batch_shape, mapping,
        leaf_log_scale, unroll,
    )
    site_likelihood = tf.reduce_sum(frequencies * partials[-1], axis=-1)
    log_scale_sum = tf.reduce_sum(log_scales, axis=0)  # leaves contribute 0
    return tf.math.log(site_likelihood) + log_scale_sum
