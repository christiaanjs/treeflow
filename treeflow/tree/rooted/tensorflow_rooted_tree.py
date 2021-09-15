import tensorflow as tf
import attr
from treeflow.tree.rooted.base_rooted_tree import AbstractRootedTree
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.tree.topology.tensorflow_tree_topology import (
    numpy_topology_to_tensor,
    TensorflowTreeTopology,
)
from treeflow.tf_util import AttrsLengthMixin
import tensorflow_probability.python.internal.prefer_static as ps


@attr.attrs(auto_attribs=True, slots=True)
class TensorflowRootedTreeAttrs(
    AbstractRootedTree[tf.Tensor, tf.Tensor], AttrsLengthMixin
):

    heights: tf.Tensor
    topology: TensorflowTreeTopology


class TensorflowRootedTree(TensorflowRootedTreeAttrs):
    @property
    def branch_lengths(self) -> tf.Tensor:
        height_shape = tf.shape(self.heights)
        indices_shape = tf.shape(self.topology.parent_indices)
        batch_shape = ps.broadcast_shape(height_shape[:-1], indices_shape[:-1])

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

    @property
    def sampling_times(self) -> tf.Tensor:
        return self.heights[..., : self.taxon_count]


def convert_tree_to_tensor(numpy_tree: NumpyRootedTree) -> TensorflowRootedTree:
    topology = numpy_topology_to_tensor(numpy_tree.topology)
    return TensorflowRootedTree(
        heights=tf.convert_to_tensor(numpy_tree.heights), topology=topology
    )


__all__ = [TensorflowRootedTree.__name__, convert_tree_to_tensor.__name__]
