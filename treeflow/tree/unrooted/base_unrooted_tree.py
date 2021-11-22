import typing as tp
from treeflow.tree.base_tree import AbstractTree
from treeflow.tree.topology.base_tree_topology import (
    AbstractTreeTopology,
    BaseTreeTopology,
)
import attr
from treeflow.tf_util import AttrsLengthMixin

TDataType = tp.TypeVar("TDataType")
TShapeType = tp.TypeVar("TShapeType")


@attr.s(auto_attribs=True, slots=True)
class BaseUnrootedTree(
    tp.Generic[TDataType],
    AttrsLengthMixin,
):
    topology: BaseTreeTopology[TDataType]
    branch_lengths: TDataType


@attr.s(auto_attribs=True, slots=True)
class AbstractUnrootedTree(
    BaseUnrootedTree,
    AbstractTree[TDataType, TShapeType],
    tp.Generic[TDataType, TShapeType],
):
    topology: AbstractTreeTopology[TDataType, TShapeType]


__all__ = [BaseUnrootedTree.__name__, AbstractUnrootedTree.__name__]
