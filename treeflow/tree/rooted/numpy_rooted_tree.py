from treeflow.tree import topology
from treeflow.tree.rooted.base_rooted_tree import AbstractRootedTree
from treeflow.tree.taxon_set import TaxonSet
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
import numpy as np
import attr
import typing as tp


@attr.attrs(auto_attribs=True)
class NumpyRootedTree(AbstractRootedTree[np.ndarray, int]):
    topology: NumpyTreeTopology
    heights: np.ndarray

    @property
    def branch_lengths(self) -> np.ndarray:
        batch_shape = np.broadcast_shapes(
            self.heights.shape[:-1], self.topology.parent_indices.shape[:-1]
        )
        heights_b = np.broadcast_to(self.heights, batch_shape + self.heights.shape[-1:])
        parent_indices_b = np.broadcast_to(
            self.topology.parent_indices,
            batch_shape + self.topology.parent_indices.shape[-1:],
        )
        return (
            np.take_along_axis(heights_b, parent_indices_b, axis=-1)
            - heights_b[..., :-1]
        )

    @property
    def sampling_times(self) -> np.ndarray:
        return self.heights[..., : self.taxon_count]
