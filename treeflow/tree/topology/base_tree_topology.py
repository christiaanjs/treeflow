import typing as tp

# from treeflow.tf_util.dict_wrapper import DictWrapper
from abc import abstractproperty

__all__ = ["TreeTopology"]

TDataType = tp.TypeVar("TDataType")
import attr


@attr.s(auto_attribs=True, slots=True)
class BaseTreeTopology(tp.Generic[TDataType]):
    parent_indices: TDataType


class AbstractTreeTopology(BaseTreeTopology[TDataType], tp.Generic[TDataType]):
    @abstractproperty
    def child_indices(self) -> TDataType:
        pass

    @abstractproperty
    def preorder_indices(self) -> TDataType:
        pass

    @abstractproperty
    def taxon_count(self) -> TDataType:
        pass
