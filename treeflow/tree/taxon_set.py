from abc import abstractmethod
import typing as tp
from typing_extensions import Protocol


class TaxonSet(Protocol):
    """
    Interface for taxon sets.
    """

    def __init__(self, taxa: tp.Iterable[str]):
        ...

    def __eq__(self, o: object) -> bool:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> tp.Iterator[str]:
        ...

    def __contains__(self, value: str):
        ...


class DictTaxonSet(tp.Dict[str, None]):
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


class TupleTaxonSet(tp.Tuple[str, ...]):
    """
    Tuple-based taxon set.
    Required for AutoCompositeTensor.
    """

    pass
