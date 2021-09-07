import typing as tp


class TaxonSet(tp.Dict[str, None]):
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
