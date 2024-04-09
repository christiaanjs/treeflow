from abc import abstractmethod
import typing as tp
from typing_extensions import Protocol


class TaxonSet(Protocol):
    """
    Interface representing a taxon set.

    A TaxonSet is an ordered set of taxon names, usually associated with
    the ordered leaves of a phylogenetic tree.
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
    Taxon set implementation based on the built-in dictionary type.

    An TaxonSet is an ordered set of taxon names, usually associated with
    the ordered leaves of a phylogenetic tree.
    The dictionary is used as an ordered set - the keys are the taxon names.
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

    def __repr__(self) -> str:
        return repr(tuple(self.keys()))

    def __str__(self) -> str:
        return str(tuple(self.keys()))


class TupleTaxonSet(tp.Tuple[str, ...]):
    """
    Taxon set implementation based on the built-in tuple type.

    An TaxonSet is an ordered set of taxon names, usually associated with
    the ordered leaves of a phylogenetic tree.
    This implementation is required for some TensorFlow functionality.
    """

    pass


__all__ = ["TaxonSet", "DictTaxonSet", "TupleTaxonSet"]
