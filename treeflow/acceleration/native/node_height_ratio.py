"""Native (C++) node-height ratio transform op with analytic autodiff.

This is a drop-in replacement for
:func:`treeflow.traversal.ratio_transform.ratios_to_node_heights` that runs the
preorder ratio-to-height traversal in a compiled TensorFlow custom op instead of
a Python-level ``tf.TensorArray`` loop.

The forward op returns the node heights. The gradient (registered with
TensorFlow's autodiff via :func:`tf.RegisterGradient`) reuses those saved
heights to compute an exact analytic gradient with respect to the ratios and the
anchor heights by a single reverse-preorder traversal -- no recomputation of the
forward pass, and no per-node Python ops.
"""
import os
import typing as tp

import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops

_LIB_NAME = "_node_height_ratio_op.so"
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

    @tf_ops.RegisterGradient("NodeHeightRatio")
    def _node_height_ratio_grad(op, grad_heights):
        ratios = op.inputs[0]
        anchor_heights = op.inputs[1]
        preorder_indices = op.inputs[2]
        parent_indices = op.inputs[3]
        heights = op.outputs[0]
        grad_ratios, grad_anchor = _module.node_height_ratio_grad(
            grad_heights,
            heights,
            ratios,
            anchor_heights,
            preorder_indices,
            parent_indices,
        )
        # Order matches op.inputs: ratios, anchor_heights, preorder_indices,
        # parent_indices.
        return [grad_ratios, grad_anchor, None, None]


def _prepare_indices(preorder_node_indices, parent_indices):
    index_dtype = preorder_node_indices.dtype
    if index_dtype not in (tf.int32, tf.int64):
        index_dtype = tf.int32
        preorder_node_indices = tf.cast(preorder_node_indices, index_dtype)
    parent_indices = tf.cast(parent_indices, index_dtype)
    return preorder_node_indices, parent_indices


def native_ratios_to_node_heights(
    preorder_node_indices: tf.Tensor,
    parent_indices: tf.Tensor,
    ratios: tf.Tensor,
    anchor_heights: tf.Tensor,
) -> tf.Tensor:
    """Node heights from height ratios, computed by the native op.

    Drop-in replacement for
    :func:`treeflow.traversal.ratio_transform.ratios_to_node_heights` (same
    positional argument order).

    Parameters
    ----------
    preorder_node_indices
        Tensor with shape ``[internal_node]`` (no batch dimensions): the
        internal node ids in preorder, with the root first. Indices are in the
        internal-node space (i.e. already offset so the root is
        ``internal_node - 1``).
    parent_indices
        Tensor with shape ``[internal_node]`` (or longer): the parent internal
        node id of each internal node. The root entry is never read.
    ratios
        Tensor with shape ``[..., internal_node]``. The last entry of the last
        axis is the (unconstrained) root height; the rest are proportions in
        ``[0, 1]``.
    anchor_heights
        Tensor with shape ``[..., internal_node]`` (broadcast against
        ``ratios``). The per-node lower bound the ratio interpolates from.

    Returns
    -------
    Tensor with shape ``broadcast(ratios, anchor_heights)`` whose last axis is
    indexed by internal node id, matching the reference implementation.
    """
    module = load_op_library()

    ratios = tf.convert_to_tensor(ratios)
    anchor_heights = tf.cast(
        tf.convert_to_tensor(anchor_heights), ratios.dtype
    )

    node_count = tf.shape(ratios)[-1:]
    ratios_batch = tf.shape(ratios)[:-1]
    anchor_batch = tf.shape(anchor_heights)[:-1]
    full_batch = tf.broadcast_dynamic_shape(ratios_batch, anchor_batch)
    full_shape = tf.concat([full_batch, node_count], axis=0)

    # The ratio/anchor vectors are small ([node]); broadcasting them to a common
    # batch and flattening is cheap, and lets TF autodiff reduce the anchor
    # gradient back through the broadcast automatically.
    ratios_b = tf.broadcast_to(ratios, full_shape)
    anchor_b = tf.broadcast_to(anchor_heights, full_shape)

    batch_size = tf.reduce_prod(full_batch)
    flat_shape = tf.concat([[batch_size], node_count], axis=0)
    ratios_flat = tf.reshape(ratios_b, flat_shape)
    anchor_flat = tf.reshape(anchor_b, flat_shape)

    preorder_node_indices, parent_indices = _prepare_indices(
        preorder_node_indices, parent_indices
    )

    heights_flat = module.node_height_ratio(
        ratios_flat, anchor_flat, preorder_node_indices, parent_indices
    )
    return tf.reshape(heights_flat, full_shape)


__all__ = [
    "native_ratios_to_node_heights",
    "load_op_library",
    "is_available",
    "library_path",
]
