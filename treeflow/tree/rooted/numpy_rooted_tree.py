from platform import node
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
    node_heights: np.ndarray
    sampling_times: np.ndarray
    topology: NumpyTreeTopology


class NumpyRootedTree(
    NumpyRootedTreeAttrs,
):
    UnrootedTreeType = NumpyUnrootedTree

    def __init__(
        self,
        heights: tp.Optional[np.ndarray] = None,
        node_heights: tp.Optional[np.ndarray] = None,
        sampling_times: tp.Optional[np.ndarray] = None,
        topology: tp.Optional[NumpyTreeTopology] = None,
        parent_indices: tp.Optional[np.ndarray] = None,
        taxon_set: tp.Optional[TaxonSet] = None,
    ):
        if topology is None:
            assert (
                parent_indices is not None
            ), "Either `topology` or `parent_indices` must be specified"
            topology = NumpyTreeTopology(
                parent_indices=parent_indices, taxon_set=taxon_set
            )
        taxon_count = topology.taxon_count
        if node_heights is not None:
            if sampling_times is None:
                final_sampling_times = np.zeros(
                    node_heights.shape[:-1] + (taxon_count,),
                    dtype=node_heights.dtype,
                )
            else:
                final_sampling_times = sampling_times

        else:
            assert heights is not None
            final_sampling_times = heights[..., :taxon_count]
            node_heights = heights[..., taxon_count:]
        super().__init__(
            node_heights=node_heights,
            sampling_times=final_sampling_times,
            topology=topology,
        )

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
    def heights(self) -> np.ndarray:
        return np.concatenate((self.sampling_times, self.node_heights), axis=-1)


__all__ = [NumpyRootedTree.__name__]
