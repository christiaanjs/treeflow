from treeflow.tree import topology
from treeflow.tree.rooted.base_rooted_tree import AbstractRootedTree
from treeflow.tree.taxon_set import TaxonSet
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
from treeflow.tree.unrooted.numpy_unrooted_tree import NumpyUnrootedTree
import numpy as np
import attr
import typing as tp


@attr.attrs(auto_attribs=True, slots=True)
class NumpyRootedTreeAttrs(
    AbstractRootedTree[np.ndarray, int, NumpyUnrootedTree]
):  # Convenience type hint
    heights: np.ndarray
    topology: NumpyTreeTopology


class NumpyRootedTree(
    NumpyRootedTreeAttrs,
):
    UnrootedTreeType = NumpyUnrootedTree

    def __init__(
        self,
        heights: np.ndarray,
        topology: tp.Optional[NumpyTreeTopology] = None,
        parent_indices: tp.Optional[np.ndarray] = None,
        taxon_set: tp.Optional[TaxonSet] = None,
    ):
        if topology is not None:
            super().__init__(heights=heights, topology=topology)
        elif parent_indices is not None:
            new_topology = NumpyTreeTopology(
                parent_indices=parent_indices, taxon_set=taxon_set
            )
            super().__init__(heights=heights, topology=new_topology)
        else:
            raise ValueError("Either `topology` or `parent_indices` must be specified")

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

    @property
    def internal_node_heights(self) -> np.ndarray:
        return self.heights[..., self.taxon_count :]


__all__ = [NumpyRootedTree.__name__]
