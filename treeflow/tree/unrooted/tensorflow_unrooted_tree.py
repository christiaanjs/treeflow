import tensorflow as tf
import attr
from treeflow.tree.unrooted.base_unrooted_tree import BaseUnrootedTree
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology


@attr.s(auto_attribs=True, slots=True)
class TensorflowUnrootedTree(BaseUnrootedTree[tf.Tensor, tf.Tensor]):
    topology: TensorflowTreeTopology
    branch_lengths: tf.Tensor
