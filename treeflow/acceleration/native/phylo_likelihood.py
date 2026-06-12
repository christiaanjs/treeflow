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
        node_partials = op.outputs[1]
        grad_probs, grad_freqs = _module.phylo_likelihood_grad(
            grad_site_likelihood,
            transition_probs,
            frequencies,
            node_partials,
            postorder_indices,
            child_indices,
        )
        # Order matches op.inputs: sequences, transition_probs, frequencies,
        # postorder_indices, child_indices.
        return [None, grad_probs, grad_freqs, None, None]

    @tf_ops.RegisterGradient("PhyloLikelihoodRescaled")
    def _phylo_likelihood_rescaled_grad(
        op, grad_site_log_likelihood, grad_node_partials, grad_node_scales
    ):
        del grad_node_partials, grad_node_scales  # internal carries
        transition_probs = op.inputs[1]
        frequencies = op.inputs[2]
        postorder_indices = op.inputs[3]
        child_indices = op.inputs[4]
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
        )
        return [None, grad_probs, grad_freqs, None, None]


def _canonicalize_batch(sequences_onehot, transition_probs, frequencies):
    """Broadcast the leading batch dims of all three tensors to a common shape.

    Returns the flattened-batch tensors expected by the op plus the original
    (un-flattened) broadcast batch shape so the output can be reshaped back.

    The op accepts a transition_probs / frequencies leading dim of either 1
    (broadcast over the batch) or the full batch size. We keep the broadcast
    fast-path (avoid materialising one transition matrix set per site) whenever
    those inputs have no genuine batch dimensions.
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

    probs_batch_size = tf.reduce_prod(probs_batch)
    if_probs_broadcast = tf.equal(probs_batch_size, 1)
    probs_b = tf.cond(
        if_probs_broadcast,
        lambda: tf.reshape(transition_probs, tf.stack([1, nodes, state, state])),
        lambda: tf.reshape(
            tf.broadcast_to(
                transition_probs,
                tf.concat([full_batch, [nodes, state, state]], axis=0),
            ),
            tf.stack([batch_size, nodes, state, state]),
        ),
    )

    freqs_batch_size = tf.reduce_prod(freqs_batch)
    if_freqs_broadcast = tf.equal(freqs_batch_size, 1)
    freqs_b = tf.cond(
        if_freqs_broadcast,
        lambda: tf.reshape(frequencies, tf.stack([1, state])),
        lambda: tf.reshape(
            tf.broadcast_to(frequencies, tf.concat([full_batch, [state]], axis=0)),
            tf.stack([batch_size, state]),
        ),
    )

    return sequences_b, probs_b, freqs_b, full_batch


def native_phylogenetic_likelihood(
    sequences_onehot: tf.Tensor,
    transition_probs: tf.Tensor,
    frequencies: tf.Tensor,
    postorder_node_indices: tf.Tensor,
    child_indices: tf.Tensor,
    batch_shape=(),
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
    """
    del batch_shape  # inferred from inputs
    module = load_op_library()

    sequences_b, probs_b, freqs_b, full_batch = _canonicalize_batch(
        sequences_onehot, transition_probs, frequencies
    )
    postorder_node_indices, child_indices = _prepare_indices(
        postorder_node_indices, child_indices
    )

    site_likelihood, _ = module.phylo_likelihood(
        sequences_b,
        probs_b,
        freqs_b,
        postorder_node_indices,
        child_indices,
    )
    return tf.reshape(site_likelihood, full_batch)


def _prepare_indices(postorder_node_indices, child_indices):
    index_dtype = postorder_node_indices.dtype
    if index_dtype not in (tf.int32, tf.int64):
        postorder_node_indices = tf.cast(postorder_node_indices, tf.int32)
        child_indices = tf.cast(child_indices, tf.int32)
    else:
        child_indices = tf.cast(child_indices, index_dtype)
    return postorder_node_indices, child_indices


def native_phylogenetic_log_likelihood_rescaled(
    sequences_onehot: tf.Tensor,
    transition_probs: tf.Tensor,
    frequencies: tf.Tensor,
    postorder_node_indices: tf.Tensor,
    child_indices: tf.Tensor,
    batch_shape=(),
) -> tf.Tensor:
    """Numerically stable per-site phylogenetic LOG likelihood (native op).

    Same inputs as :func:`native_phylogenetic_likelihood`, but rescales the
    partial likelihoods at every internal node (dividing by their per-site
    maximum and accumulating the log scale factors) so deep/large trees do not
    underflow. Returns the per-site ``log`` likelihood (shape = batch shape),
    not the linear likelihood.
    """
    del batch_shape  # inferred from inputs
    module = load_op_library()

    sequences_b, probs_b, freqs_b, full_batch = _canonicalize_batch(
        sequences_onehot, transition_probs, frequencies
    )
    postorder_node_indices, child_indices = _prepare_indices(
        postorder_node_indices, child_indices
    )

    site_log_likelihood, _, _ = module.phylo_likelihood_rescaled(
        sequences_b,
        probs_b,
        freqs_b,
        postorder_node_indices,
        child_indices,
    )
    return tf.reshape(site_log_likelihood, full_batch)


__all__ = [
    "native_phylogenetic_likelihood",
    "native_phylogenetic_log_likelihood_rescaled",
    "load_op_library",
    "is_available",
    "library_path",
]
