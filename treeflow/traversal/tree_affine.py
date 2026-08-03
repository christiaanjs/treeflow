"""Affine tree maps: the linear layers of the tree normalising flow.

Two structured affine maps on the vector of per-internal-node coordinates
``x`` (shape ``[..., internal_node]``, indexed by internal node id exactly as
the node-height ratios are):

**Preorder (root-to-tip) affine** -- each node reads its *parent's output*::

    y[root] = scale[root] * x[root] + shift[root]
    y[i]    = scale[i] * x[i] + shift[i] + parent_weight[i] * y[parent(i)]

**Postorder (tip-to-root) affine** -- each node reads its *children's
outputs*, with leaf children contributing nothing::

    w[i] = scale[i] * x[i] + shift[i]
           + sum_{c in children(i), c internal} child_weight[i, c] * w[c]

Both are linear in ``x`` and triangular in their own traversal order, so:

* the Jacobian determinant is ``prod_i scale[i]`` -- no traversal needed to
  evaluate the log-det-Jacobian, and the map is a bijection whenever every
  ``scale[i] != 0`` (the flow parameterises ``scale`` as a softplus, so it is
  positive by construction);
* the *inverse* needs no traversal either. Inverting the preorder map only
  requires each node's parent's ``y``, and inverting the postorder map only
  requires each node's children's ``w`` -- both already available in the
  quantity being inverted, so a single ``tf.gather`` suffices. Only the
  forward direction is an actual sequential traversal (mirroring
  :class:`~treeflow.bijectors.node_height_ratio_bijector.NodeHeightRatioBijector`,
  whose native acceleration is likewise forward-only).

Composing the two around a nonlinearity (see
:mod:`treeflow.bijectors.tree_normalizing_flow`) gives a map in which *any*
node's coordinate can influence any other: the preorder half propagates
information down from the root, the postorder half pulls it back up from the
tips, and neither alone can do both.

The node axis is the **last** axis throughout, matching
:mod:`treeflow.traversal.ratio_transform` and the unconstrained tree
coordinates the flow sits underneath.
"""

import typing as tp

import tensorflow as tf
import tensorflow_probability.python.internal.prefer_static as ps

from treeflow.traversal.postorder import postorder_node_traversal
from treeflow.traversal.preorder import preorder_traversal
from treeflow.traversal.ratio_transform import (
    _node_axis_to_front,
    move_outside_axis_to_inside as _node_axis_to_back,
)
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology


def _batch_shape(x: tf.Tensor, event_ndims: int):
    return ps.shape(x)[: ps.rank(x) - event_ndims]


def _broadcast_batch(
    tensors_and_event_ndims: tp.Sequence[tp.Tuple[tf.Tensor, int]]
) -> tp.List[tf.Tensor]:
    """Broadcast a set of tensors to a common batch shape, keeping each one's
    own trailing event dimensions."""
    batch = _batch_shape(tensors_and_event_ndims[0][0], tensors_and_event_ndims[0][1])
    for tensor, event_ndims in tensors_and_event_ndims[1:]:
        batch = ps.broadcast_shape(batch, _batch_shape(tensor, event_ndims))
    return [
        tf.broadcast_to(
            tensor,
            ps.concat([batch, ps.shape(tensor)[ps.rank(tensor) - event_ndims :]], axis=0),
        )
        for tensor, event_ndims in tensors_and_event_ndims
    ]


def node_parent_indices(topology: TensorflowTreeTopology) -> tf.Tensor:
    """Parent of each **non-root** internal node, in internal-node space.

    Shape ``[internal_node - 1]``: treeflow's ``parent_indices`` drops the root,
    so slicing off the leaves leaves one entry per non-root internal node (the
    same quantity ``NodeHeightRatioBijector`` inverts with).
    """
    taxon_count = topology.taxon_count
    return topology.parent_indices[taxon_count:] - taxon_count


def node_child_indices(topology: TensorflowTreeTopology) -> tf.Tensor:
    """Children of each internal node, in internal-node space.

    Shape ``[internal_node, child]``. Leaf children come out **negative** (a
    leaf's id is below ``taxon_count``), which is how both the pure-TensorFlow
    and the native implementations recognise them.
    """
    taxon_count = topology.taxon_count
    return topology.node_child_indices - taxon_count


def preorder_affine(
    topology: TensorflowTreeTopology,
    x: tf.Tensor,
    scale: tf.Tensor,
    shift: tf.Tensor,
    parent_weight: tf.Tensor,
    unroll: tp.Union[bool, str] = "auto",
) -> tf.Tensor:
    """Root-to-tip affine map, on the generic :func:`preorder_traversal`.

    Parameters
    ----------
    x, scale, shift, parent_weight
        Tensors with shape ``[..., internal_node]`` (broadcasting against each
        other). ``parent_weight``'s root entry (the last) is unused -- the root
        has no parent -- but is accepted so every per-node parameter block has
        the same shape.
    unroll
        Forwarded to :func:`~treeflow.traversal.preorder.preorder_traversal`.
    """
    x, scale, shift, parent_weight = _broadcast_batch(
        [(x, 1), (scale, 1), (shift, 1), (parent_weight, 1)]
    )
    x_nf = _node_axis_to_front(x)
    scale_nf = _node_axis_to_front(scale)
    shift_nf = _node_axis_to_front(shift)
    weight_nf = _node_axis_to_front(parent_weight)

    root_init = scale_nf[-1] * x_nf[-1] + shift_nf[-1]

    def mapping(parent_output, node_input):
        scale_i, shift_i, weight_i, x_i = node_input
        return scale_i * x_i + shift_i + weight_i * parent_output

    y_nf = preorder_traversal(
        topology,
        mapping,
        (scale_nf, shift_nf, weight_nf, x_nf),
        root_init,
        unroll=unroll,
    )
    return _node_axis_to_back(y_nf)


def preorder_affine_inverse(
    topology: TensorflowTreeTopology,
    y: tf.Tensor,
    scale: tf.Tensor,
    shift: tf.Tensor,
    parent_weight: tf.Tensor,
) -> tf.Tensor:
    """Inverse of :func:`preorder_affine` -- a gather, not a traversal.

    Each node's parent's output is part of ``y``, so every coordinate can be
    undone independently::

        x[i] = (y[i] - shift[i] - parent_weight[i] * y[parent(i)]) / scale[i]
    """
    parent_indices = node_parent_indices(topology)
    parent_y = tf.gather(y, parent_indices, axis=-1)
    nonroot = (
        y[..., :-1] - shift[..., :-1] - parent_weight[..., :-1] * parent_y
    ) / scale[..., :-1]
    root = (y[..., -1:] - shift[..., -1:]) / scale[..., -1:]
    return tf.concat([nonroot, root], axis=-1)


def postorder_affine(
    topology: TensorflowTreeTopology,
    x: tf.Tensor,
    scale: tf.Tensor,
    shift: tf.Tensor,
    child_weight: tf.Tensor,
    unroll: tp.Union[bool, str] = "auto",
) -> tf.Tensor:
    """Tip-to-root affine map, on the generic :func:`postorder_node_traversal`.

    Parameters
    ----------
    x, scale, shift
        Tensors with shape ``[..., internal_node]``.
    child_weight
        Tensor with shape ``[..., internal_node, child]``. Entries for leaf
        children are unused (a leaf carries no flow coordinate); the traversal
        seeds the leaves with exact zeros so they drop out of the sum.
    unroll
        Forwarded to
        :func:`~treeflow.traversal.postorder.postorder_node_traversal`.
    """
    x, scale, shift, child_weight = _broadcast_batch(
        [(x, 1), (scale, 1), (shift, 1), (child_weight, 2)]
    )
    x_nf = _node_axis_to_front(x)
    scale_nf = _node_axis_to_front(scale)
    shift_nf = _node_axis_to_front(shift)
    # [..., node, child] -> [node, child, ...]: the traversal stacks the child
    # outputs on axis 0, so the child axis must sit directly behind the node one.
    weight_rank = ps.rank(child_weight)
    weight_nf = tf.transpose(
        child_weight,
        ps.concat([[weight_rank - 2, weight_rank - 1], ps.range(weight_rank - 2)], axis=0),
    )

    taxon_count = topology.taxon_count
    batch_shape = ps.shape(x)[:-1]
    leaf_init = tf.zeros(ps.concat([[taxon_count], batch_shape], axis=0), dtype=x.dtype)

    def mapping(child_output, node_input, topology_data):
        del topology_data
        scale_i, shift_i, weight_i, x_i = node_input
        return (
            scale_i * x_i
            + shift_i
            + tf.reduce_sum(weight_i * child_output, axis=0)
        )

    all_nf = postorder_node_traversal(
        topology,
        mapping,
        (scale_nf, shift_nf, weight_nf, x_nf),
        leaf_init,
        unroll=unroll,
    )
    return _node_axis_to_back(all_nf[taxon_count:])


def postorder_affine_inverse(
    topology: TensorflowTreeTopology,
    y: tf.Tensor,
    scale: tf.Tensor,
    shift: tf.Tensor,
    child_weight: tf.Tensor,
) -> tf.Tensor:
    """Inverse of :func:`postorder_affine` -- a gather, not a traversal.

    Each node's children's outputs are part of ``y``, so::

        x[i] = (y[i] - shift[i] - sum_c child_weight[i, c] * y[c]) / scale[i]
    """
    child_indices = node_child_indices(topology)  # [node, child], leaves negative
    internal_mask = tf.cast(child_indices >= 0, y.dtype)
    safe_indices = tf.maximum(child_indices, 0)
    child_y = tf.gather(y, safe_indices, axis=-1)  # [..., node, child]
    child_sum = tf.reduce_sum(child_weight * internal_mask * child_y, axis=-1)
    return (y - shift - child_sum) / scale


def affine_log_det_jacobian(scale: tf.Tensor) -> tf.Tensor:
    """Log-det-Jacobian of either affine map: ``sum_i log scale[i]``.

    Both maps are triangular in their own traversal order with ``scale`` on the
    diagonal, so no traversal is involved.
    """
    return tf.reduce_sum(tf.math.log(scale), axis=-1)


__all__ = [
    "preorder_affine",
    "preorder_affine_inverse",
    "postorder_affine",
    "postorder_affine_inverse",
    "affine_log_det_jacobian",
    "node_parent_indices",
    "node_child_indices",
]
