import typing as tp


class TaxonSet(tp.Dict[str, None]):
    def __init__(
        self, *dict_args, taxa: tp.Optional[tp.Iterable[str]] = None, **dict_kwargs
    ):
        if taxa is not None:
            super().__init__([(taxon, None) for taxon in taxa])
        else:
            super().__init__(*dict_args, **dict_kwargs)
