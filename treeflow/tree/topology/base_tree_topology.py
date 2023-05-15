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
    @abstractproperty
    def child_indices(self) -> TDataType:
        pass

    @abstractproperty
    def preorder_indices(self) -> TDataType:
        pass

    @abstractproperty
    def taxon_count(self) -> TShapeType:
        pass

    @property
    def postorder_node_indices(self) -> TDataType:
        pass

    @abstractproperty
    def taxon_set(self) -> tp.Optional[TaxonSet]:
        pass


__all__ = [BaseTreeTopology.__name__, AbstractTreeTopology.__name__]
