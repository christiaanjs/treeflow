from treeflow.tree import taxon_set
import typing as tp
import attr
from treeflow.tree.topology.base_tree_topology import AbstractTreeTopology
import numpy as np
from treeflow.tree.taxon_set import TaxonSet
from treeflow.tree.topology.numpy_topology_operations import (
    get_child_indices,
    get_preorder_indices,
)


@attr.attrs(auto_attribs=True)
class NumpyTreeTopologyAttrs(AbstractTreeTopology[np.ndarray, int]):
    parent_indices: np.ndarray  # Convenience type hint


class NumpyTreeTopology(NumpyTreeTopologyAttrs):
    """
    Class representing a bifurcating tree topology as a composition of integer
    NumPy arrays.

    For a phylogenetic tree with ``n`` taxa at the leaves, the representation
    maintains a labelling of the ``1n-1`` nodes with integer indices. The labelling
    convention is that the leaves are the first ``n`` indices and the root is at the
    last index (``2n-2``).
    """

    def __init__(
        self, parent_indices: np.ndarray, taxon_set: tp.Optional[TaxonSet] = None
    ):
        super().__init__(parent_indices=parent_indices)
        self._taxon_set = taxon_set

    @property
    def taxon_count(self) -> int:
        return (self.parent_indices.shape[-1] + 2) // 2

    @property
    def node_count(self) -> int:
        return 2 * self.taxon_count - 1

    @property
    def postorder_node_indices(self) -> np.ndarray:
        return np.arange(self.taxon_count, 2 * self.taxon_count - 1)

    @property
    def child_indices(self) -> np.ndarray:
        return get_child_indices(self.parent_indices)

    @property
    def preorder_indices(self) -> np.ndarray:
        return get_preorder_indices(self.child_indices)

    @property
    def taxon_set(self) -> tp.Optional[TaxonSet]:
        return self._taxon_set

    # Methods to allow pickling
    def __getstate__(self):
        return super().__getstate__() + (self._taxon_set,)

    def __setstate__(self, state):
        super().__setstate__(state[:-1])
        self._taxon_set = state[-1]


__all__ = ["NumpyTreeTopology"]
