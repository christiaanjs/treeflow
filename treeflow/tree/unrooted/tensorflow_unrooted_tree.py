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
