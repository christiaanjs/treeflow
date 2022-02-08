from __future__ import annotations

import tensorflow as tf
import numpy as np
import attr
from tensorflow.python.framework.ops import to_raw_op
from treeflow.tree.rooted.base_rooted_tree import AbstractRootedTree
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.tree.topology.tensorflow_tree_topology import (
    numpy_topology_to_tensor,
    TensorflowTreeTopology,
)
from treeflow.tree.unrooted.tensorflow_unrooted_tree import TensorflowUnrootedTree
import tensorflow_probability.python.internal.prefer_static as ps
import typing as tp
from treeflow.tree.taxon_set import TaxonSet
from treeflow import DEFAULT_FLOAT_DTYPE_TF


@attr.attrs(auto_attribs=True, slots=True)
class TensorflowRootedTreeAttrs(
    AbstractRootedTree[tf.Tensor, tf.Tensor, TensorflowUnrootedTree]
):

    node_heights: tf.Tensor
    sampling_times: tf.Tensor
    topology: TensorflowTreeTopology


class TensorflowRootedTree(TensorflowRootedTreeAttrs):
    UnrootedTreeType = TensorflowUnrootedTree

    @property
    def heights(self) -> tf.Tensor:
        batch_shape = tf.shape(self.node_heights)[:-1]
        sampling_time_shape = tf.concat(
            [batch_shape, tf.shape(self.sampling_times)], axis=0
        )
        sampling_times = tf.broadcast_to(self.sampling_times, sampling_time_shape)
        return tf.concat((sampling_times, self.node_heights), axis=-1)

    @property
    def branch_lengths(self) -> tf.Tensor:
        height_shape = tf.shape(self.heights)
        indices_shape = tf.shape(self.topology.parent_indices)
        batch_shape = ps.broadcast_shape(height_shape[:-1], indices_shape[:-1])
        # Implement with tf_util
        heights_b = tf.broadcast_to(
            self.heights, tf.concat([batch_shape, height_shape[-1:]], 0)
        )
        parent_indices_b = tf.broadcast_to(
            self.topology.parent_indices,
            tf.concat([batch_shape, indices_shape[-1:]], 0),
        )
        batch_shape = tf.shape(batch_shape)[-1]
        return (
            tf.gather(heights_b, parent_indices_b, batch_dims=-1) - heights_b[..., :-1]
        )

    def numpy(self) -> NumpyRootedTree:
        return NumpyRootedTree(
            node_heights=self.node_heights.numpy(),
            sampling_times=self.sampling_times.numpy(),
            topology=self.topology.numpy(),
        )

    def with_node_heights(self, node_heights: tf.Tensor) -> TensorflowRootedTree:
        return TensorflowRootedTree(
            node_heights=node_heights,
            topology=self.topology,
            sampling_times=self.sampling_times,
        )


def convert_tree_to_tensor(
    numpy_tree: NumpyRootedTree, height_dtype: tf.DType = DEFAULT_FLOAT_DTYPE_TF
) -> TensorflowRootedTree:
    topology = numpy_topology_to_tensor(numpy_tree.topology)
    return TensorflowRootedTree(
        sampling_times=tf.convert_to_tensor(
            numpy_tree.sampling_times, dtype=height_dtype
        ),
        node_heights=tf.convert_to_tensor(numpy_tree.node_heights, dtype=height_dtype),
        topology=topology,
    )


def tree_from_arrays(
    node_heights: tp.Union[np.ndarray, tf.Tensor],
    parent_indices: tp.Union[np.ndarray, tf.Tensor],
    sampling_times: tp.Optional[tp.Union[np.ndarray, tf.Tensor]],
    taxon_set: tp.Optional[TaxonSet] = None,
    height_dtype: tf.DType = DEFAULT_FLOAT_DTYPE_TF,
) -> TensorflowRootedTree:

    node_heights_np = (
        node_heights.numpy() if isinstance(node_heights, tf.Tensor) else node_heights
    )
    sampling_times_np = (
        sampling_times.numpy()
        if isinstance(sampling_times, tf.Tensor)
        else sampling_times
    )
    parent_indices_np = (
        parent_indices.numpy()
        if isinstance(parent_indices, tf.Tensor)
        else parent_indices
    )

    numpy_tree = NumpyRootedTree(
        sampling_times=sampling_times_np,
        node_heights=node_heights_np,
        parent_indices=parent_indices_np,
        taxon_set=taxon_set,
    )
    return convert_tree_to_tensor(numpy_tree, height_dtype=height_dtype)


__all__ = [
    TensorflowRootedTree.__name__,
    convert_tree_to_tensor.__name__,
    tree_from_arrays.__name__,
]
