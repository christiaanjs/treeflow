import numpy as np
import attr
from treeflow.tree.unrooted.base_unrooted_tree import AbstractUnrootedTree
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology


@attr.s(auto_attribs=True, slots=True)
class NumpyUnrootedTree(AbstractUnrootedTree[np.ndarray, int]):
    topology: NumpyTreeTopology
    branch_lengths: np.ndarray
