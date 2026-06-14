"""Native (C++) phylogenetic likelihood op with analytic autodiff.

This is a drop-in replacement for
:func:`treeflow.traversal.phylo_likelihood.phylogenetic_likelihood` that runs
Felsenstein's pruning algorithm in a compiled TensorFlow custom op instead of a
Python-level ``tf.TensorArray`` loop.

The forward op returns the per-site likelihoods together with the partial
likelihood vectors at every node. The gradient (registered with TensorFlow's
autodiff via :func:`tf.RegisterGradient`) reuses those saved partials to
compute an exact analytic gradient with respect to the transition probability
matrices and the root frequencies -- no recomputation of the forward
traversal, and no per-node Python ops.
"""
import os
import typing as tp

import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops

_LIB_NAME = "_phylo_likelihood_op.so"
_module: tp.Optional[tp.Any] = None


def library_path() -> str:
    return os.path.join(os.path.dirname(__file__), _LIB_NAME)


def load_op_library():
    """Load (once) and return the compiled op library module."""
    global _module
    if _module is None:
        path = library_path()
        if not os.path.exists(path):
            raise RuntimeError(
                f"Native op library not found at {path}. "
                "Build it with treeflow/acceleration/native/build.sh "
                "(or `python -m treeflow.acceleration.native.build`)."
            )
        _module = tf.load_op_library(path)
        _register_gradient()
    return _module


def is_available() -> bool:
    """Return True if the native op library is built and loadable."""
    try:
        load_op_library()
        return True
    except Exception:
        return False


_gradient_registered = False


def _register_gradient():
    global _gradient_registered
    if _gradient_registered:
        return
    _gradient_registered = True

    @tf_ops.RegisterGradient("PhyloLikelihood")
    def _phylo_likelihood_grad(op, grad_site_likelihood, grad_node_partials):
        # grad_node_partials is unused: node_partials is an internal carry used
        # only to feed the backward op, so no gradient flows back through it.
        del grad_node_partials
        transition_probs = op.inputs[1]
        frequencies = op.inputs[2]
        postorder_indices = op.inputs[3]
        child_indices = op.inputs[4]
        probs_index = op.inputs[5]
        freqs_index = op.inputs[6]
        node_partials = op.outputs[1]
        grad_probs, grad_freqs = _module.phylo_likelihood_grad(
            grad_site_likelihood,
            transition_probs,
            frequencies,
            node_partials,
            postorder_indices,
            child_indices,
            probs_index,
            freqs_index,
            block_size=op.get_attr("block_size"),
        )
        # Order matches op.inputs: sequences, transition_probs, frequencies,
        # postorder_indices, child_indices, probs_index, freqs_index.
        return [None, grad_probs, grad_freqs, None, None, None, None]

    @tf_ops.RegisterGradient("PhyloLikelihoodRescaled")
    def _phylo_likelihood_rescaled_grad(
        op, grad_site_log_likelihood, grad_node_partials, grad_node_scales
    ):
        del grad_node_partials, grad_node_scales  # internal carries
        transition_probs = op.inputs[1]
        frequencies = op.inputs[2]
        postorder_indices = op.inputs[3]
        child_indices = op.inputs[4]
        probs_index = op.inputs[5]
        freqs_index = op.inputs[6]
        node_partials = op.outputs[1]
        node_scales = op.outputs[2]
        grad_probs, grad_freqs = _module.phylo_likelihood_rescaled_grad(
            grad_site_log_likelihood,
            transition_probs,
            frequencies,
            node_partials,
            node_scales,
            postorder_indices,
            child_indices,
            probs_index,
            freqs_index,
            block_size=op.get_attr("block_size"),
        )
        return [None, grad_probs, grad_freqs, None, None, None, None]


def _broadcast_gather_index(batch_shape, batch_size, full_batch, full_size):
    """Per-(flattened) batch-element index into a tensor's own flattened batch.

    Maps each position of the broadcast ``full_batch`` to the flattened index of
    the element it reads from a tensor whose batch shape is ``batch_shape`` (which
    must be broadcastable to ``full_batch``). This lets the native op *gather* the
    transition matrices / frequencies for each batch element instead of
    materialising a broadcast (tiled) copy -- e.g. for a discrete rate-category
    mixture the transition matrices only vary across the ``M`` categories, so we
    keep ``[M, node, state, state]`` and index it, rather than tiling to
    ``[sites * M, node, state, state]``.

    Works for any broadcasting pattern (leading sample dims, interleaved
    category/parameter-sample dims, etc.): ``reshape(range(n), batch_shape)``
    labels each batch element with its flattened index, and broadcasting that
    label tensor to ``full_batch`` reads off the gather index per position.
    """
    labels = tf.reshape(tf.range(batch_size), batch_shape)
    return tf.reshape(tf.broadcast_to(labels, full_batch), tf.stack([full_size]))


def _canonicalize_batch(sequences_onehot, transition_probs, frequencies):
    """Flatten the batch dims and build per-element gather indices for the op.

    Returns the flattened-batch tensors expected by the op, the gather indices
    that locate each batch element's transition matrices / frequencies, and the
    original (un-flattened) broadcast batch shape so the output can be reshaped
    back.

    The transition matrices and frequencies keep their *own* (un-tiled) batch
    size ``Bt`` / ``Bf``; the op reads ``transition_probs[probs_index[b]]`` and
    ``frequencies[freqs_index[b]]`` for batch element ``b``. ``Bt`` / ``Bf`` may
    be 1 (broadcast over the whole batch), the full batch size (a distinct matrix
    set per element), or anything in between (e.g. one set per rate category).
    """
    seq_batch = tf.shape(sequences_onehot)[:-2]
    probs_batch = tf.shape(transition_probs)[:-3]
    freqs_batch = tf.shape(frequencies)[:-1]

    full_batch = tf.broadcast_dynamic_shape(
        tf.broadcast_dynamic_shape(seq_batch, probs_batch), freqs_batch
    )
    batch_size = tf.reduce_prod(full_batch)

    state = tf.shape(sequences_onehot)[-1]
    leaves = tf.shape(sequences_onehot)[-2]
    nodes = tf.shape(transition_probs)[-3]

    sequences_b = tf.reshape(
        tf.broadcast_to(
            sequences_onehot,
            tf.concat([full_batch, [leaves, state]], axis=0),
        ),
        tf.stack([batch_size, leaves, state]),
    )

    # Transition matrices and frequencies are kept at their own batch size (no
    # tiling); the op gathers per batch element via the index tensors below.
    probs_batch_size = tf.reduce_prod(probs_batch)
    probs_b = tf.reshape(
        transition_probs, tf.stack([probs_batch_size, nodes, state, state])
    )
    probs_index = _broadcast_gather_index(
        probs_batch, probs_batch_size, full_batch, batch_size
    )

    freqs_batch_size = tf.reduce_prod(freqs_batch)
    freqs_b = tf.reshape(frequencies, tf.stack([freqs_batch_size, state]))
    freqs_index = _broadcast_gather_index(
        freqs_batch, freqs_batch_size, full_batch, batch_size
    )

    return sequences_b, probs_b, freqs_b, probs_index, freqs_index, full_batch


def native_phylogenetic_likelihood(
    sequences_onehot: tf.Tensor,
    transition_probs: tf.Tensor,
    frequencies: tf.Tensor,
    postorder_node_indices: tf.Tensor,
    child_indices: tf.Tensor,
    batch_shape=(),
    block_size: int = 1,
) -> tf.Tensor:
    """Per-site phylogenetic likelihood, computed by the native op.

    Drop-in replacement for
    :func:`treeflow.traversal.phylo_likelihood.phylogenetic_likelihood`.

    Parameters mirror the reference implementation:

    sequences_onehot
        Tensor with shape ``[..., leaf, state]``.
    transition_probs
        Tensor with shape ``[..., node, state, state]``. Broadcast over sites.
    frequencies
        Tensor with shape ``[..., state]``.
    postorder_node_indices
        Tensor with shape ``[internal_node]`` (no batch dimensions). Internal
        node ids in postorder.
    child_indices
        Tensor with shape ``[internal_node, children]``.
    batch_shape
        Unused; accepted for signature compatibility with the reference
        implementation (the batch shape is inferred from the inputs).
    block_size
        Number of alignment sites processed together so the per-node
        transition-matrix products vectorise over the site dimension (SIMD).
        ``1`` (default) processes sites individually, reproducing the original
        traversal. Larger values (e.g. 8/16/32) enable site-blocking. The
        result is bit-identical regardless of ``block_size``; this only affects
        performance.
    """
    del batch_shape  # inferred from inputs
    module = load_op_library()

    sequences_b, probs_b, freqs_b, probs_index, freqs_index, full_batch = (
        _canonicalize_batch(sequences_onehot, transition_probs, frequencies)
    )
    postorder_node_indices, child_indices, probs_index, freqs_index = _prepare_indices(
        postorder_node_indices, child_indices, probs_index, freqs_index
    )

    site_likelihood, _ = module.phylo_likelihood(
        sequences_b,
        probs_b,
        freqs_b,
        postorder_node_indices,
        child_indices,
        probs_index,
        freqs_index,
        block_size=block_size,
    )
    return tf.reshape(site_likelihood, full_batch)


def _prepare_indices(postorder_node_indices, child_indices, probs_index, freqs_index):
    index_dtype = postorder_node_indices.dtype
    if index_dtype not in (tf.int32, tf.int64):
        index_dtype = tf.int32
        postorder_node_indices = tf.cast(postorder_node_indices, index_dtype)
    child_indices = tf.cast(child_indices, index_dtype)
    # The batch gather indices must share the op's Tindex type.
    probs_index = tf.cast(probs_index, index_dtype)
    freqs_index = tf.cast(freqs_index, index_dtype)
    return postorder_node_indices, child_indices, probs_index, freqs_index


def native_phylogenetic_log_likelihood_rescaled(
    sequences_onehot: tf.Tensor,
    transition_probs: tf.Tensor,
    frequencies: tf.Tensor,
    postorder_node_indices: tf.Tensor,
    child_indices: tf.Tensor,
    batch_shape=(),
    block_size: int = 1,
) -> tf.Tensor:
    """Numerically stable per-site phylogenetic LOG likelihood (native op).

    Same inputs as :func:`native_phylogenetic_likelihood`, but rescales the
    partial likelihoods at every internal node (dividing by their per-site
    maximum and accumulating the log scale factors) so deep/large trees do not
    underflow. Returns the per-site ``log`` likelihood (shape = batch shape),
    not the linear likelihood.

    ``block_size`` controls site-blocking exactly as in
    :func:`native_phylogenetic_likelihood`.
    """
    del batch_shape  # inferred from inputs
    module = load_op_library()

    sequences_b, probs_b, freqs_b, probs_index, freqs_index, full_batch = (
        _canonicalize_batch(sequences_onehot, transition_probs, frequencies)
    )
    postorder_node_indices, child_indices, probs_index, freqs_index = _prepare_indices(
        postorder_node_indices, child_indices, probs_index, freqs_index
    )

    site_log_likelihood, _, _ = module.phylo_likelihood_rescaled(
        sequences_b,
        probs_b,
        freqs_b,
        postorder_node_indices,
        child_indices,
        probs_index,
        freqs_index,
        block_size=block_size,
    )
    return tf.reshape(site_log_likelihood, full_batch)


__all__ = [
    "native_phylogenetic_likelihood",
    "native_phylogenetic_log_likelihood_rescaled",
    "load_op_library",
    "is_available",
    "library_path",
]
