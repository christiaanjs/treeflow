"""TFP bijectors for the affine tree maps of the tree normalising flow.

Thin wrappers around :mod:`treeflow.traversal.tree_affine` (and its native C++
counterpart) that present each structured affine map as a
:class:`~tensorflow_probability.python.bijectors.Bijector` on the per-node
coordinate vector, so they can be chained with the elementwise flow layer --
and with anything else in the TFP bijector framework.

Both maps are triangular in their own traversal order with ``scale`` on the
diagonal, so the log-det-Jacobian is ``sum_i log scale[i]``: constant in the
coordinates, and free to evaluate. Only the forward direction runs a traversal;
the inverse is a gather (see the module docstring of
:mod:`treeflow.traversal.tree_affine`).

Like :class:`~treeflow.bijectors.node_height_ratio_bijector.NodeHeightRatioBijector`,
the forward traversal can be run either by the pure-TensorFlow traversal
primitives or by the native op (``use_native``); the native op registers a
first-order gradient only, so code needing higher-order derivatives through the
forward map should pass ``use_native=False``.
"""

import typing as tp

import tensorflow as tf
from tensorflow_probability.python.bijectors import Bijector

from treeflow.traversal.tree_affine import (
    affine_log_det_jacobian,
    node_child_indices,
    node_parent_indices,
    postorder_affine,
    postorder_affine_inverse,
    preorder_affine,
    preorder_affine_inverse,
)
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology


def native_tree_affine_available() -> bool:
    """Return True if the native affine tree map ops can be loaded."""
    try:
        from treeflow.acceleration.native import tree_affine_is_available

        return tree_affine_is_available()
    except Exception:
        return False


def _resolve_use_native(use_native: tp.Union[bool, str]) -> bool:
    if use_native == "auto":
        return native_tree_affine_available()
    if isinstance(use_native, bool):
        return use_native
    raise ValueError(f"use_native must be True, False, or 'auto'; got {use_native!r}")


class _TreeAffineBijector(Bijector):
    """Shared plumbing for the two affine tree maps."""

    def __init__(
        self,
        topology: TensorflowTreeTopology,
        scale: tf.Tensor,
        shift: tf.Tensor,
        weight: tf.Tensor,
        use_native: tp.Union[bool, str] = "auto",
        unroll: tp.Union[bool, str] = "auto",
        name: str = "TreeAffineBijector",
        validate_args: bool = False,
    ):
        parameters = dict(locals())
        self.topology = topology
        self.scale = scale
        self.shift = shift
        self.weight = weight
        self.unroll = unroll
        self.use_native = use_native
        self._use_native = _resolve_use_native(use_native)
        super().__init__(
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
            dtype=scale.dtype,
            validate_args=validate_args,
            parameters=parameters,
            name=name,
        )

    def _forward_log_det_jacobian(self, x):
        del x  # linear map: the Jacobian does not depend on the coordinates
        return affine_log_det_jacobian(self.scale)

    def _inverse_log_det_jacobian(self, y):
        del y
        return -affine_log_det_jacobian(self.scale)


class PreorderAffineBijector(_TreeAffineBijector):
    """Root-to-tip affine map::

        y[root] = scale[root] * x[root] + shift[root]
        y[i]    = scale[i] * x[i] + shift[i] + parent_weight[i] * y[parent(i)]

    ``parent_weight``'s root entry is unused (the root has no parent) but is
    accepted so all per-node parameter blocks share a shape.
    """

    def __init__(
        self,
        topology: TensorflowTreeTopology,
        scale: tf.Tensor,
        shift: tf.Tensor,
        parent_weight: tf.Tensor,
        use_native: tp.Union[bool, str] = "auto",
        unroll: tp.Union[bool, str] = "auto",
        name: str = "PreorderAffineBijector",
        validate_args: bool = False,
    ):
        super().__init__(
            topology,
            scale,
            shift,
            parent_weight,
            use_native=use_native,
            unroll=unroll,
            name=name,
            validate_args=validate_args,
        )

    def _forward(self, x):
        if self._use_native:
            from treeflow.acceleration.native import native_preorder_affine

            taxon_count = self.topology.taxon_count
            return native_preorder_affine(
                self.topology.preorder_node_indices - taxon_count,
                node_parent_indices(self.topology),
                x,
                self.scale,
                self.shift,
                self.weight,
            )
        return preorder_affine(
            self.topology, x, self.scale, self.shift, self.weight, unroll=self.unroll
        )

    def _inverse(self, y):
        return preorder_affine_inverse(
            self.topology, y, self.scale, self.shift, self.weight
        )


class PostorderAffineBijector(_TreeAffineBijector):
    """Tip-to-root affine map::

        w[i] = scale[i] * x[i] + shift[i]
               + sum_{c internal} child_weight[i, c] * w[c]

    ``child_weight`` has shape ``[..., internal_node, child]``; entries for leaf
    children are unused, as leaves carry no flow coordinate.
    """

    def __init__(
        self,
        topology: TensorflowTreeTopology,
        scale: tf.Tensor,
        shift: tf.Tensor,
        child_weight: tf.Tensor,
        use_native: tp.Union[bool, str] = "auto",
        unroll: tp.Union[bool, str] = "auto",
        name: str = "PostorderAffineBijector",
        validate_args: bool = False,
    ):
        super().__init__(
            topology,
            scale,
            shift,
            child_weight,
            use_native=use_native,
            unroll=unroll,
            name=name,
            validate_args=validate_args,
        )

    def _forward(self, x):
        if self._use_native:
            from treeflow.acceleration.native import native_postorder_affine

            taxon_count = self.topology.taxon_count
            return native_postorder_affine(
                self.topology.postorder_node_indices - taxon_count,
                node_child_indices(self.topology),
                x,
                self.scale,
                self.shift,
                self.weight,
            )
        return postorder_affine(
            self.topology, x, self.scale, self.shift, self.weight, unroll=self.unroll
        )

    def _inverse(self, y):
        return postorder_affine_inverse(
            self.topology, y, self.scale, self.shift, self.weight
        )


__all__ = [
    "PreorderAffineBijector",
    "PostorderAffineBijector",
    "native_tree_affine_available",
]
