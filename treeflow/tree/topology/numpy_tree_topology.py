import attr
from treeflow.tree.topology.base_tree_topology import AbstractTreeTopology
import numpy as np
from numpy.typing import ArrayLike
from treeflow.tree.topology.numpy_topology_operations import (
    get_child_indices,
    get_preorder_indices,
)


@attr.attrs(auto_attribs=True)
class NumpyTreeTopology(AbstractTreeTopology[np.ndarray]):
    parent_indices: np.ndarray  # Convenience type hint

    @property
    def postorder_node_indices(self) -> np.ndarray:
        return np.arange(self.taxon_count, 2 * self.taxon_count - 1)

    @property
    def child_indices(self) -> np.ndarray:
        return get_child_indices(self.parent_indices)

    @property
    def preorder_indices(self) -> np.ndarray:
        return get_preorder_indices(self.child_indices)
