from abc import abstractproperty
from treeflow.tree.base_tree import AbstractTree
from treeflow.tree.topology.base_tree_topology import (
    AbstractTreeTopology,
    BaseTreeTopology,
)
import attr
import typing as tp
from treeflow.tree.taxon_set import TaxonSet

TDataType = tp.TypeVar("TDataType")
TShapeType = tp.TypeVar("TShapeType")


@attr.s(auto_attribs=True, slots=True)
class BaseRootedTree(tp.Generic[TDataType]):
    topology: BaseTreeTopology[TDataType]
    heights: TDataType


@attr.s(auto_attribs=True, slots=True)
class AbstractRootedTree(
    BaseRootedTree[TDataType],
    AbstractTree[TDataType, TShapeType],
    tp.Generic[TDataType, TShapeType],
):
    topology: AbstractTreeTopology[TDataType, TShapeType]  # Repeat for type hint
    heights: TDataType

    @abstractproperty
    def sampling_times(self) -> TDataType:
        pass
