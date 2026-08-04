"""Native (C++) affine tree maps with analytic autodiff.

Drop-in replacements for :func:`treeflow.traversal.tree_affine.preorder_affine`
and :func:`treeflow.traversal.tree_affine.postorder_affine` that run the
sequential sweep in a compiled TensorFlow custom op instead of a Python-level
unrolled/``tf.TensorArray`` traversal. These are the linear layers of the tree
normalising flow, so they are evaluated once per flow layer per gradient step.

The forward ops return the transformed coordinates. The gradients (registered
with TensorFlow's autodiff via :func:`tf.RegisterGradient`) reuse those saved
outputs to compute exact analytic gradients with respect to the coordinates and
every per-node parameter in a single reverse sweep -- no recomputation of the
forward pass, and no per-node Python ops.

Only the forward direction is compiled: both maps invert with a single
``tf.gather`` (see the module docstring of
:mod:`treeflow.traversal.tree_affine`), and their log-det-Jacobian is
``sum_i log scale[i]``.
"""
import os
import typing as tp

import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops

_LIB_NAME = "_tree_affine_op.so"
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
        _register_gradients()
    return _module


def is_available() -> bool:
    """Return True if the native op library is built and loadable."""
    try:
        load_op_library()
        return True
    except Exception:
        return False


_gradients_registered = False


def _register_gradients():
    global _gradients_registered
    if _gradients_registered:
        return
    _gradients_registered = True

    @tf_ops.RegisterGradient("TreeAffinePreorder")
    def _tree_affine_preorder_grad(op, grad_y):
        x, scale, shift, parent_weight, preorder_indices, parent_indices = op.inputs
        y = op.outputs[0]
        grads = _module.tree_affine_preorder_grad(
            grad_y,
            y,
            x,
            scale,
            shift,
            parent_weight,
            preorder_indices,
            parent_indices,
        )
        # Order matches op.inputs; the index tensors are non-differentiable.
        return list(grads) + [None, None]

    @tf_ops.RegisterGradient("TreeAffinePostorder")
    def _tree_affine_postorder_grad(op, grad_w):
        x, scale, shift, child_weight, postorder_indices, child_indices = op.inputs
        w = op.outputs[0]
        grads = _module.tree_affine_postorder_grad(
            grad_w,
            w,
            x,
            scale,
            shift,
            child_weight,
            postorder_indices,
            child_indices,
        )
        return list(grads) + [None, None]


def _prepare_indices(*index_tensors):
    """Cast a set of index tensors to a common integer dtype the op accepts."""
    dtype = index_tensors[0].dtype
    if dtype not in (tf.int32, tf.int64):
        dtype = tf.int32
    return [tf.cast(tensor, dtype) for tensor in index_tensors]


def _broadcast_and_flatten(tensors_and_event_ndims, node_count):
    """Broadcast to a common batch shape and flatten it to a single axis.

    Every parameter is small (one or two entries per node), so broadcasting and
    flattening is cheap, and it lets TF autodiff reduce a shared (unbatched)
    parameter's gradient back through the broadcast automatically.
    """
    batch_shapes = [
        tf.shape(tensor)[: tf.rank(tensor) - event_ndims]
        for tensor, event_ndims in tensors_and_event_ndims
    ]
    full_batch = batch_shapes[0]
    for batch_shape in batch_shapes[1:]:
        full_batch = tf.broadcast_dynamic_shape(full_batch, batch_shape)
    batch_size = tf.reduce_prod(full_batch)

    flat = []
    for tensor, event_ndims in tensors_and_event_ndims:
        event_shape = [node_count] if event_ndims == 1 else [node_count, tf.shape(tensor)[-1]]
        broadcast_shape = tf.concat([full_batch, event_shape], axis=0)
        flat_shape = tf.concat([[batch_size], event_shape], axis=0)
        flat.append(tf.reshape(tf.broadcast_to(tensor, broadcast_shape), flat_shape))
    return flat, full_batch


def native_preorder_affine(
    preorder_node_indices: tf.Tensor,
    parent_indices: tf.Tensor,
    x: tf.Tensor,
    scale: tf.Tensor,
    shift: tf.Tensor,
    parent_weight: tf.Tensor,
) -> tf.Tensor:
    """Root-to-tip affine map, computed by the native op.

    Parameters
    ----------
    preorder_node_indices
        Tensor with shape ``[internal_node]`` (no batch dimensions): the
        internal node ids in preorder, root first, in internal-node space.
    parent_indices
        Tensor with shape ``[internal_node - 1]``: the parent internal node id
        of each non-root internal node.
    x, scale, shift, parent_weight
        Tensors with shape ``[..., internal_node]``, broadcasting against each
        other. ``parent_weight``'s root entry is unused (and gets a zero
        gradient).

    Returns
    -------
    Tensor with the broadcast shape of the inputs.
    """
    module = load_op_library()

    x = tf.convert_to_tensor(x)
    scale = tf.cast(tf.convert_to_tensor(scale), x.dtype)
    shift = tf.cast(tf.convert_to_tensor(shift), x.dtype)
    parent_weight = tf.cast(tf.convert_to_tensor(parent_weight), x.dtype)

    node_count = tf.shape(x)[-1]
    (x_f, scale_f, shift_f, weight_f), full_batch = _broadcast_and_flatten(
        [(x, 1), (scale, 1), (shift, 1), (parent_weight, 1)], node_count
    )

    preorder_node_indices, parent_indices = _prepare_indices(
        preorder_node_indices, parent_indices
    )
    y_flat = module.tree_affine_preorder(
        x_f, scale_f, shift_f, weight_f, preorder_node_indices, parent_indices
    )
    return tf.reshape(y_flat, tf.concat([full_batch, [node_count]], axis=0))


def native_postorder_affine(
    postorder_node_indices: tf.Tensor,
    child_indices: tf.Tensor,
    x: tf.Tensor,
    scale: tf.Tensor,
    shift: tf.Tensor,
    child_weight: tf.Tensor,
) -> tf.Tensor:
    """Tip-to-root affine map, computed by the native op.

    Parameters
    ----------
    postorder_node_indices
        Tensor with shape ``[internal_node]`` (no batch dimensions): the
        internal node ids in postorder, root last, in internal-node space.
    child_indices
        Tensor with shape ``[internal_node, child]``: child ids in
        internal-node space, **negative** for leaf children (which carry no
        coordinate and whose weights are unused).
    x, scale, shift
        Tensors with shape ``[..., internal_node]``.
    child_weight
        Tensor with shape ``[..., internal_node, child]``.

    Returns
    -------
    Tensor with the broadcast shape of ``x``, ``scale`` and ``shift`` (and
    ``child_weight``'s batch).
    """
    module = load_op_library()

    x = tf.convert_to_tensor(x)
    scale = tf.cast(tf.convert_to_tensor(scale), x.dtype)
    shift = tf.cast(tf.convert_to_tensor(shift), x.dtype)
    child_weight = tf.cast(tf.convert_to_tensor(child_weight), x.dtype)

    node_count = tf.shape(x)[-1]
    (x_f, scale_f, shift_f, weight_f), full_batch = _broadcast_and_flatten(
        [(x, 1), (scale, 1), (shift, 1), (child_weight, 2)], node_count
    )

    postorder_node_indices, child_indices = _prepare_indices(
        postorder_node_indices, child_indices
    )
    w_flat = module.tree_affine_postorder(
        x_f, scale_f, shift_f, weight_f, postorder_node_indices, child_indices
    )
    return tf.reshape(w_flat, tf.concat([full_batch, [node_count]], axis=0))


__all__ = [
    "native_preorder_affine",
    "native_postorder_affine",
    "load_op_library",
    "is_available",
    "library_path",
]
