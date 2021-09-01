from treeflow.tree.topology.base_tree_topology import AbstractTreeTopology
import numpy as np
from numpy.typing import ArrayLike


class NumpyTreeTopology(AbstractTreeTopology[np.ndarray]):
    def child_indices(self) -> np.ndarray:
        
        