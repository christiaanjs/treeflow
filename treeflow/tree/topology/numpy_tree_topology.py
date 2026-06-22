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
        return (super().__getstate__(), self._taxon_set)

    def __setstate__(self, state):
        super().__setstate__(state[0])
        self._taxon_set = state[1]


class StaticNumpyTreeTopology:
    """A ``NumpyTreeTopology`` variant with every derived index array pre-computed at
    construction.

    ``NumpyTreeTopology`` is an ``attrs`` class, which ``tf.nest`` expands into its
    single ``parent_indices`` leaf. Passing one into a ``tf.function`` therefore
    replaces ``parent_indices`` with a symbolic placeholder, and the lazily-computed
    NumPy properties (``child_indices`` etc.) then fail on the symbolic tensor.

    This class is a plain Python object instead, so ``tf.nest`` treats it as an
    *atom*: it threads through traced code as a captured constant, and its index
    arrays stay concrete NumPy arrays inside the trace. That lets ``tf.get_static_value``
    fold the topology regardless of tree size, so the node-height ratio-transform
    traversal unrolls rather than falling back to a ``tf.while_loop``.

    It is meant to be used as a fixed-topology *pin*. It is not a Tensor, so it is
    not placed directly into a JointDistribution's tree value (which TFP coerces to
    tensors); ``FixedTopologyRootedTreeBijector`` rebuilds it as an in-graph-constant
    ``TensorflowTreeTopology`` for that purpose (see its ``_forward``).
    """

    def __init__(
        self, parent_indices: np.ndarray, taxon_set: tp.Optional[TaxonSet] = None
    ):
        parent_indices = np.asarray(parent_indices, dtype=np.int32)
        child_indices = np.asarray(get_child_indices(parent_indices), dtype=np.int32)
        preorder_indices = np.asarray(
            get_preorder_indices(child_indices), dtype=np.int32
        )
        taxon_count = int((parent_indices.shape[-1] + 2) // 2)
        self._parent_indices = parent_indices
        self._child_indices = child_indices
        self._preorder_indices = preorder_indices
        self._taxon_count = taxon_count
        self._node_count = 2 * taxon_count - 1
        self._postorder_node_indices = np.arange(
            taxon_count, self._node_count, dtype=np.int32
        )
        self._node_child_indices = child_indices[taxon_count:]
        self._preorder_node_indices = preorder_indices[preorder_indices >= taxon_count]
        self._taxon_set = taxon_set

    @classmethod
    def from_numpy_topology(
        cls, topology: NumpyTreeTopology
    ) -> "StaticNumpyTreeTopology":
        return cls(topology.parent_indices, taxon_set=topology.taxon_set)

    @property
    def parent_indices(self) -> np.ndarray:
        return self._parent_indices

    @property
    def child_indices(self) -> np.ndarray:
        return self._child_indices

    @property
    def preorder_indices(self) -> np.ndarray:
        return self._preorder_indices

    @property
    def taxon_count(self) -> int:
        return self._taxon_count

    @property
    def node_count(self) -> int:
        return self._node_count

    @property
    def postorder_node_indices(self) -> np.ndarray:
        return self._postorder_node_indices

    @property
    def node_child_indices(self) -> np.ndarray:
        return self._node_child_indices

    @property
    def preorder_node_indices(self) -> np.ndarray:
        return self._preorder_node_indices

    @property
    def taxon_set(self) -> tp.Optional[TaxonSet]:
        return self._taxon_set

    def has_batch_dimensions(self) -> bool:
        return False

    def to_constant_tensor_topology(self):
        """Return a ``TensorflowTreeTopology`` whose index arrays are ``tf.constant``s.

        IMPORTANT: call this *inside* a ``tf.function`` trace (e.g. a VI loss step,
        the fixed-topology bijector's ``forward``, or a profiler timing function).
        The arrays then become in-graph ``Const`` ops, which gives both properties we
        need at once:

        * a normal tensor topology that a ``JointDistribution`` accepts in its tree
          value (it coerces the value to tensors), and
        * a topology that still folds via ``tf.get_static_value`` at *any* tree size,
          so the downstream traversals unroll.

        Calling it eagerly and capturing the result into a later trace loses the
        second property: a captured constant can be re-materialised as a
        ``Placeholder`` for large trees (the >64-taxon non-fold), which is exactly
        what building the ``Const`` inside the trace avoids.
        """
        import tensorflow as tf
        from treeflow.tree.topology.tensorflow_tree_topology import (
            TensorflowTreeTopology,
        )

        return TensorflowTreeTopology(
            parent_indices=tf.constant(self._parent_indices),
            child_indices=tf.constant(self._child_indices),
            preorder_indices=tf.constant(self._preorder_indices),
        )

    def numpy(self) -> NumpyTreeTopology:
        return NumpyTreeTopology(
            parent_indices=self._parent_indices, taxon_set=self._taxon_set
        )


__all__ = ["NumpyTreeTopology", "StaticNumpyTreeTopology"]
