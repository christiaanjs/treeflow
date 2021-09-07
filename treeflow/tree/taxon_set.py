from abc import abstractmethod
import typing as tp
from collections.abc import Set
from typing_extensions import Protocol


class TaxonSet(Protocol, tp.Collection[str]):
    """
    Interface for taxon sets.
    """

    def __eq__(self, o: object) -> bool:
        pass


class DictTaxonSet(tp.Dict[str, None], TaxonSet):
    """
    Dict-based taxon set.
    Dictionary is used an ordered set (of keys)
    """

    def __init__(self, taxa: tp.Iterable[str]):
        super().__init__([(taxon, None) for taxon in taxa])

    def __eq__(self, o: object) -> bool:
        if isinstance(o, tp.Iterable):
            return tuple(self) == tuple(o)
        else:
            return False

    def __ne__(self, o: object) -> bool:
        if isinstance(o, tp.Iterable):
            return tuple(self) != tuple(o)
        else:
            return True


class TupleTaxonSet(tp.Tuple[str, ...], TaxonSet):
    """
    Tuple-based taxon set.
    Required for AutoCompositeTensor.
    """

    pass
