from __future__ import annotations
import tensorflow as tf
import attr
from treeflow.tree.unrooted.base_unrooted_tree import AbstractUnrootedTree
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology


@attr.s(auto_attribs=True, slots=True)
class TensorflowUnrootedTree(AbstractUnrootedTree[tf.Tensor, tf.Tensor]):
    topology: TensorflowTreeTopology
    branch_lengths: tf.Tensor

    def with_branch_lengths(self, branch_lengths):
        return TensorflowUnrootedTree(
            topology=self.topology, branch_lengths=branch_lengths
        )

    def __mul__(self, other: tf.Tensor) -> TensorflowUnrootedTree:
        return self.with_branch_lengths(self.branch_lengths * other)

    def __rmul__(self, other: tf.Tensor) -> TensorflowUnrootedTree:
        return self.__mul__(other)
