from abc import abstractproperty
from treeflow.tree.base_tree import AbstractTree
from treeflow.tree.topology.base_tree_topology import (
    AbstractTreeTopology,
    BaseTreeTopology,
)
from treeflow.tree.unrooted.base_unrooted_tree import BaseUnrootedTree
import attr
import typing as tp
from treeflow.tree.taxon_set import TaxonSet
from treeflow.tf_util import AttrsLengthMixin

TDataType = tp.TypeVar("TDataType")
TShapeType = tp.TypeVar("TShapeType")
TRootedTreeType = tp.TypeVar("TRootedTreeType")


@attr.s(auto_attribs=True, slots=True)
class BaseRootedTree(tp.Generic[TDataType], AttrsLengthMixin):
    topology: BaseTreeTopology[TDataType]
    heights: TDataType


@attr.s(auto_attribs=True, slots=True)
class AbstractRootedTreeAttrs(
    BaseRootedTree[TDataType],
    AbstractTree[TDataType, TShapeType],
    tp.Generic[TDataType, TShapeType],
):
    topology: AbstractTreeTopology[TDataType, TShapeType]


class AbstractRootedTree(
    AbstractRootedTreeAttrs[TDataType, TShapeType],
    tp.Generic[TDataType, TShapeType, TRootedTreeType],
):

    UnrootedTreeType: tp.Type[TRootedTreeType] = BaseUnrootedTree

    @abstractproperty
    def sampling_times(self) -> TDataType:
        pass

    def unrooted_tree(self) -> TRootedTreeType:
        return type(self).UnrootedTreeType(
            topology=self.topology, branch_lengths=self.branch_lengths
        )
