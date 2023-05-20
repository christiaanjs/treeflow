import typing as tp

from abc import abstractproperty
from treeflow.tree.taxon_set import TaxonSet

TDataType = tp.TypeVar("TDataType")
TShapeType = tp.TypeVar("TShapeType")
import attr


@attr.s(auto_attribs=True, slots=True)
class BaseTreeTopology(tp.Generic[TDataType]):
    parent_indices: TDataType


class AbstractTreeTopology(
    BaseTreeTopology[TDataType], tp.Generic[TDataType, TShapeType]
):
    """
    Class representing a bifurcating tree topology as a composition of integer arrays.

    For a phylogenetic tree with ``n`` taxa at the leaves, the representation
    maintains a labelling of the ``1n-1`` nodes with integer indices. The labelling
    convention is that the leaves are the first ``n`` indices and the root is at the last
    index (``2n-2``).
    """

    @abstractproperty
    def child_indices(self) -> TDataType:
        """
        Array of length ``2n-2`` representing the parent-child structure of a
        tree topology on ``n`` taxa.

        The ``i`` th element of this array is the index of the parent of the ``i`` th
        indexed node in the tree, for every node except the root.
        """
        pass

    @abstractproperty
    def preorder_indices(self) -> TDataType:
        """
        Array of indices of length ``2n-1`` that form a pre-order traversal
        of the tree.

        A pre-order traversal is an ordering of the nodes where every
        node is visited before its children, starting at the root.
        """
        pass

    @abstractproperty
    def taxon_count(self) -> TShapeType:
        """Number of leaf taxa the tree is based on (``n``)"""
        pass

    @property
    def postorder_node_indices(self) -> TDataType:
        pass

    @abstractproperty
    def taxon_set(self) -> tp.Optional[TaxonSet]:
        pass


__all__ = [BaseTreeTopology.__name__, AbstractTreeTopology.__name__]
