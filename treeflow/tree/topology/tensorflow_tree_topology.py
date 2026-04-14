from __future__ import annotations

import typing as tp
import attr
import numpy as np
import tensorflow as tf
from treeflow.tree.taxon_set import TaxonSet, TupleTaxonSet
from treeflow.tree.topology.base_tree_topology import (
    AbstractTreeTopology,
    BaseTreeTopology,
)
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
import treeflow.tree.topology.numpy_topology_operations as np_top_ops
import tensorflow_probability.python.internal.prefer_static as ps
from treeflow.tf_util import AttrsLengthMixin


def tensor_taxon_count(parent_indices: tf.Tensor) -> tf.Tensor:
    parent_index_count = ps.shape(parent_indices)[-1]
    return ps.cast((parent_index_count + 2) // 2, parent_index_count.dtype)


@attr.attrs(auto_attribs=True, slots=True)
class TensorflowTreeTopologyAttrs(
    AbstractTreeTopology[tf.Tensor, tf.Tensor], AttrsLengthMixin
):
    parent_indices: tf.Tensor  # Convenience type hint
    child_indices: tf.Tensor
    preorder_indices: tf.Tensor

    @property
    def taxon_count(self) -> tf.Tensor:
        return tensor_taxon_count(self.parent_indices)

    @property
    def postorder_node_indices(self) -> tf.Tensor:
        taxon_count = self.taxon_count
        return tf.range(taxon_count, 2 * taxon_count - 1, dtype=taxon_count.dtype)

    @property
    def node_child_indices(self) -> tf.Tensor:
        return self.child_indices[self.taxon_count :]

    @property
    def preorder_node_indices(self) -> tf.Tensor:
        preorder_indices = self.preorder_indices
        return tf.boolean_mask(
            preorder_indices,
            preorder_indices >= self.taxon_count,
            axis=tf.rank(preorder_indices) - 1,
        )


class TensorflowTreeTopology(TensorflowTreeTopologyAttrs):
    def __init__(
        self,
        parent_indices: tf.Tensor,
        child_indices: tf.Tensor = None,
        preorder_indices: tf.Tensor = None,
        taxon_set: tp.Optional[TaxonSet] = None,
    ):
        self._taxon_set = (
            None if taxon_set is None else TupleTaxonSet(taxon_set)
        )  # Must be Tuple for auto composite
        super().__init__(
            parent_indices=parent_indices,
            child_indices=child_indices,
            preorder_indices=preorder_indices,
        )

    @property
    def taxon_set(self) -> tp.Optional[TaxonSet]:
        return self._taxon_set

    @classmethod
    def get_event_ndims(cls) -> TensorflowTreeTopology:
        return TensorflowTreeTopology(
            parent_indices=1, child_indices=2, preorder_indices=1
        )

    def get_prefer_static_rank(
        self,
    ) -> TensorflowTreeTopology:
        return tf.nest.map_structure(ps.rank, self)

    @classmethod
    def rank_to_has_batch_dimensions(cls, rank: TensorflowTreeTopology):
        event_ndims = cls.get_event_ndims()
        batch_ndims = tf.nest.map_structure(
            lambda elem_rank, elem_event_ndims: elem_rank - elem_event_ndims,
            rank,
            event_ndims,
        )
        has_batch_dims_array = ps.stack(tf.nest.flatten(batch_ndims)) > 0
        return ps.reduce_any(has_batch_dims_array)

    def has_batch_dimensions(self) -> tp.Union[bool, tf.Tensor]:
        rank = self.get_prefer_static_rank()
        return type(self).rank_to_has_batch_dimensions(rank)

    def numpy(self) -> NumpyTreeTopology:
        return NumpyTreeTopology(
            parent_indices=self.parent_indices, taxon_set=self.taxon_set
        )

    # Methods to allow pickling
    def __getstate__(self):
        super_state = super().__getstate__()
        return (super_state, self._taxon_set)

    def __setstate__(self, state):
        super().__setstate__(state[0])
        self._taxon_set = state[1]


def compute_preorder_indices(child_indices: tf.Tensor) -> tf.Tensor:
    """
    Compute preorder (DFS) traversal indices from child_indices, in pure TF.

    Implements an iterative pre-order DFS: the root is pushed onto a stack
    first; at each step the top node is popped and recorded; for internal
    nodes the right child is pushed then the left child so that the left
    child is visited first.

    Parameters
    ----------
    child_indices : tf.Tensor
        Shape ``[..., node_count, 2]``.  Leaf nodes have child value ``-1``.
        ``node_count`` (``child_indices.shape[-2]``) must be statically known.

    Returns
    -------
    tf.Tensor
        Preorder indices, shape ``[..., node_count]``, dtype ``tf.int32``.
    """
    static_nc = child_indices.shape[-2]
    if static_nc is None:
        raise ValueError(
            "node_count (child_indices.shape[-2]) must be statically known"
        )
    node_count = int(static_nc)
    root = node_count - 1

    unbatched = child_indices.shape.rank == 2
    if unbatched:
        child_indices = tf.expand_dims(child_indices, 0)

    n_total = tf.shape(child_indices)[0]  # may be a dynamic tensor
    nc_tf = tf.constant(node_count, dtype=tf.int32)
    neg1 = tf.constant(-1, dtype=tf.int32)

    s_idx = tf.range(n_total, dtype=tf.int32)
    zeros = tf.zeros_like(s_idx)
    ones = tf.ones_like(s_idx)

    # Initialise DFS stack with root at position 0
    stack = tf.fill(tf.stack([n_total, nc_tf]), neg1)
    stack = tf.tensor_scatter_nd_update(
        stack,
        tf.stack([s_idx, zeros], axis=1),
        tf.fill(tf.expand_dims(n_total, 0), tf.constant(root, dtype=tf.int32)),
    )
    stack_size = tf.ones_like(s_idx)
    preorder = tf.zeros(tf.stack([n_total, nc_tf]), dtype=tf.int32)
    preorder_pos = tf.zeros_like(s_idx)

    # Exactly node_count iterations suffice: each visit pops 1, internal nodes
    # push 2 children, giving net 0 for internal and -1 for leaves; the tree
    # has n leaves and n-1 internal nodes so the stack empties after node_count steps.
    for _ in range(node_count):
        # Pop top of stack
        top_pos = stack_size - 1
        current_node = tf.gather_nd(stack, tf.stack([s_idx, top_pos], axis=1))
        stack_size = stack_size - 1

        # Record current node in preorder output
        preorder = tf.tensor_scatter_nd_update(
            preorder,
            tf.stack([s_idx, preorder_pos], axis=1),
            current_node,
        )
        preorder_pos = preorder_pos + 1

        # Gather children of current node
        child0 = tf.gather_nd(
            child_indices, tf.stack([s_idx, current_node, zeros], axis=1)
        )
        child1 = tf.gather_nd(
            child_indices, tf.stack([s_idx, current_node, ones], axis=1)
        )
        is_internal = tf.not_equal(child0, neg1)

        # Push right child (child1) first so left child (child0) is popped first
        push1_pos = stack_size
        cur1 = tf.gather_nd(stack, tf.stack([s_idx, push1_pos], axis=1))
        stack = tf.tensor_scatter_nd_update(
            stack,
            tf.stack([s_idx, push1_pos], axis=1),
            tf.where(is_internal, child1, cur1),
        )
        stack_size = tf.where(is_internal, stack_size + 1, stack_size)

        # Push left child (child0)
        push0_pos = stack_size
        cur0 = tf.gather_nd(stack, tf.stack([s_idx, push0_pos], axis=1))
        stack = tf.tensor_scatter_nd_update(
            stack,
            tf.stack([s_idx, push0_pos], axis=1),
            tf.where(is_internal, child0, cur0),
        )
        stack_size = tf.where(is_internal, stack_size + 1, stack_size)

    if unbatched:
        return preorder[0]
    return preorder


def numpy_topology_to_tensor(
    topology: NumpyTreeTopology, dtype=tf.int32
) -> TensorflowTreeTopology:
    parent_indices = topology.parent_indices
    child_indices = np_top_ops.get_child_indices(parent_indices)
    child_indices_tf = tf.constant(child_indices, dtype=dtype)
    preorder_indices = compute_preorder_indices(child_indices_tf)
    return TensorflowTreeTopology(
        parent_indices=tf.constant(parent_indices, dtype=dtype),
        child_indices=child_indices_tf,
        preorder_indices=preorder_indices,
        taxon_set=(
            None if topology.taxon_set is None else TupleTaxonSet(topology.taxon_set)
        ),
    )


__all__ = [
    tensor_taxon_count.__name__,
    TensorflowTreeTopology.__name__,
    compute_preorder_indices.__name__,
]
