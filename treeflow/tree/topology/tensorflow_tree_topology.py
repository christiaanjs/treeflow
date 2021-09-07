import typing as tp
import attr
import tensorflow as tf
from treeflow.tree.taxon_set import TaxonSet, TupleTaxonSet
from treeflow.tree.topology.base_tree_topology import AbstractTreeTopology
import tensorflow_probability.python.internal.prefer_static as ps
from tensorflow_probability.python.internal.tensor_util import convert_nonref_to_tensor
from tensorflow_probability.python.experimental import (
    AutoCompositeTensor,
    auto_composite_tensor,
)
import treeflow.tree.topology.numpy_topology_operations as np_topology_operations


@attr.attrs(auto_attribs=True)
class TensorflowTreeTopologyAttrs(AbstractTreeTopology[tf.Tensor, tf.Tensor]):
    parent_indices: tf.Tensor  # Convenience type hint


@auto_composite_tensor
class TensorflowStaticTreeTopology(TensorflowTreeTopologyAttrs, AutoCompositeTensor):
    def __init__(
        self,
        parent_indices: tf.Tensor,
        taxon_set: tp.Optional[TaxonSet] = None,
        child_indices: tp.Optional[tf.Tensor] = None,
        preorder_indices: tp.Optional[tf.Tensor] = None,
        postorder_node_indices: tp.Optional[tf.Tensor] = None,
    ):
        super().__init__(parent_indices=convert_nonref_to_tensor(parent_indices))
        self._taxon_set = (
            None if taxon_set is None else TupleTaxonSet(taxon_set)
        )  # Must be Tuple for auto composite

        self._child_indices = child_indices
        self._preorder_indices = preorder_indices
        self._postorder_node_indices = postorder_node_indices

    @property
    def child_indices(self) -> tf.Tensor:
        if self._child_indices is None:
            self._child_indices = tf.convert_to_tensor(
                np_topology_operations.get_child_indices(self.parent_indices.numpy())
            )
        return self._child_indices

    @property
    def preorder_indices(self) -> tf.Tensor:
        if self.preorder_indices is None:
            self._preorder_indices = tf.convert_to_tensor(
                np_topology_operations.get_preorder_indices(self.child_indices.numpy())
            )
        return self._preorder_indices

    @property
    def taxon_count(self) -> tf.Tensor:
        return (ps.shape(self.parent_indices) + 2) // 2

    @property
    def postorder_node_indices(self) -> tf.Tensor:
        taxon_count = self.taxon_count
        return tf.range(taxon_count, 2 * taxon_count - 1)

    @property
    def taxon_set(self) -> tp.Optional[TaxonSet]:
        return self._taxon_set
