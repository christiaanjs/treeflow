import typing as tp
from treeflow.distributions.tree.birthdeath.birth_death_contemporary_sampling import (
    BirthDeathContemporarySampling,
)


class Yule(BirthDeathContemporarySampling):
    def __init__(
        self,
        taxon_count,
        birth_rate,
        validate_args=False,
        allow_nan_stats=True,
        name="Yule",
        tree_name: tp.Optional[str] = None,
    ):
        params = dict(locals())
        super().__init__(
            taxon_count=taxon_count,
            birth_diff_rate=birth_rate,
            relative_death_rate=0.0,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            tree_name=tree_name,
        )
        self._parameters = params

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        super_pp = BirthDeathContemporarySampling._parameter_properties(
            dtype, num_classes=num_classes
        )
        return dict(birth_rate=super_pp["birth_diff_rate"])


__all__ = ["Yule"]
