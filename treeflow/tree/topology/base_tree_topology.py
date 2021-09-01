import typing as tp
from treeflow.tf_util.dict_wrapper import DictWrapper
from abc import abstractproperty

__all__ = ["TreeTopology"]

TDataType = tp.TypeVar("TDataType")


class BaseTreeTopology(DictWrapper[str, TDataType], tp.Generic[TDataType]):
    PARENT_INDICES_KEY = "parent_indices"

    class_keys = set([PARENT_INDICES_KEY])

    def __init__(
        self,
        mapping: tp.Optional[tp.Mapping[str, TDataType]] = None,
        parent_indices: tp.Optional[TDataType] = None,
    ):
        super().__init__(mapping, {BaseTreeTopology.PARENT_INDICES_KEY: parent_indices})

    @property
    def parent_indices(self):
        return self[BaseTreeTopology.PARENT_INDICES_KEY]


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
