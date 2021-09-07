from abc import abstractproperty
from treeflow.tree.taxon_set import TaxonSet
import typing as tp
from treeflow.tree.topology.base_tree_topology import (
    AbstractTreeTopology,
)


TDataType = tp.TypeVar("TDataType")
TShapeType = tp.TypeVar("TShapeType")


class AbstractTree(tp.Generic[TDataType, TShapeType]):
    @abstractproperty
    def topology(self) -> AbstractTreeTopology[TDataType, TShapeType]:
        pass

    @abstractproperty
    def branch_lengths(self) -> TDataType:
        pass

    def taxon_set(self) -> tp.Optional[TaxonSet]:
        return self.topology.taxon_set

    def taxon_count(self) -> TShapeType:
        return self.topology.taxon_count
