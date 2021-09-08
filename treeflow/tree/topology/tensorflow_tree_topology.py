from types import prepare_class
import typing as tp
import attr
import tensorflow as tf
from treeflow.tree.taxon_set import TaxonSet, TupleTaxonSet
from treeflow.tree.topology.base_tree_topology import AbstractTreeTopology
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
import treeflow.tree.topology.numpy_topology_operations as np_top_ops
import tensorflow_probability.python.internal.prefer_static as ps


def tensor_taxon_count(parent_indices: tf.Tensor) -> tf.Tensor:
    return (ps.shape(parent_indices)[-1] + 2) // 2


@attr.attrs(auto_attribs=True, slots=True)
class TensorflowTreeTopologyAttrs(AbstractTreeTopology[tf.Tensor, tf.Tensor]):
    parent_indices: tf.Tensor  # Convenience type hint
    child_indices: tf.Tensor
    preorder_indices: tf.Tensor

    @property
    def taxon_count(self) -> tf.Tensor:
        return tensor_taxon_count(self.parent_indices)

    @property
    def postorder_node_indices(self) -> tf.Tensor:
        taxon_count = self.taxon_count
        return tf.range(taxon_count, 2 * taxon_count - 1)


class TensorflowTreeTopology(TensorflowTreeTopologyAttrs):
    def __init__(
        self,
        parent_indices: tf.Tensor,
        child_indices: tf.Tensor = None,
        preorder_indices: tf.Tensor = None,
        taxon_set: tp.Optional[TaxonSet] = None,
    ):
        self._taxon_set = (
            None if taxon_set is None else TupleTaxonSet(taxon_set)
        )  # Must be Tuple for auto composite
        super().__init__(
            parent_indices=parent_indices,
            child_indices=child_indices,
            preorder_indices=preorder_indices,
        )

    @property
    def taxon_set(self) -> tp.Optional[TaxonSet]:
        return self._taxon_set


def numpy_topology_to_tensor(topology: NumpyTreeTopology) -> TensorflowTreeTopology:
    parent_indices = topology.parent_indices
    child_indices = np_top_ops.get_child_indices(parent_indices)
    preorder_indices = np_top_ops.get_preorder_indices(child_indices)
    return TensorflowTreeTopology(
        parent_indices=tf.convert_to_tensor(parent_indices),
        child_indices=tf.convert_to_tensor(child_indices),
        preorder_indices=tf.convert_to_tensor(preorder_indices),
        taxon_set=(
            None if topology.taxon_set is None else TupleTaxonSet(topology.taxon_set)
        ),
    )


__all__ = [
    tensor_taxon_count.__name__,
    TensorflowTreeTopology.__name__,
]
