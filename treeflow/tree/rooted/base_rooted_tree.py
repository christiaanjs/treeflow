from __future__ import annotations

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
    node_heights: TDataType
    sampling_times: TDataType


@attr.s(auto_attribs=True, slots=True, init=False)
class AbstractRootedTreeAttrs(
    BaseRootedTree[TDataType],
    AbstractTree[TDataType, TShapeType],
    tp.Generic[TDataType, TShapeType],
):
    topology: AbstractTreeTopology[TDataType, TShapeType]

    def __init__(
        self,
        tree_or_first_arg: tp.Optional[
            tp.Union[AbstractRootedTreeAttrs, object]
        ] = None,
        *args,
        **kwargs,
    ):  # This logic is because `tf.nest.cast_structure` expects a copy constructor
        if isinstance(tree_or_first_arg, BaseRootedTree):
            self.__attrs_init__(
                topology=tree_or_first_arg.topology,
                node_heights=tree_or_first_arg.node_heights,
                sampling_times=tree_or_first_arg.sampling_times,
            )
        elif attr.fields(type(self))[0].name in kwargs:
            self.__attrs_init__(*args, **kwargs)
        else:
            self.__attrs_init__(tree_or_first_arg, *args, **kwargs)


class AbstractRootedTree(
    AbstractRootedTreeAttrs[TDataType, TShapeType],
    tp.Generic[TDataType, TShapeType, TRootedTreeType],
):

    UnrootedTreeType: tp.Type = BaseUnrootedTree  # TODO: Better type hint

    @abstractproperty
    def heights(self) -> TDataType:
        pass

    def get_unrooted_tree(self) -> TRootedTreeType:
        return type(self).UnrootedTreeType(
            topology=self.topology, branch_lengths=self.branch_lengths
        )
