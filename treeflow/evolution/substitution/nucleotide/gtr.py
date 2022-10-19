from struct import pack
import tensorflow as tf
from treeflow.evolution.substitution.eigendecomposition import Eigendecomposition
from treeflow.evolution.substitution.base_substitution_model import (
    EigendecompositionSubstitutionModel,
)
from treeflow.evolution.substitution.util import pack_matrix

GTR_RATE_ORDER = ("ac", "ag", "at", "cg", "ct", "gt")


class GTR(EigendecompositionSubstitutionModel):
    def q(self, frequencies: tf.Tensor, rates: tf.Tensor) -> tf.Tensor:
        pi = frequencies
        return pack_matrix(
            [
                [
                    -(
                        rates[..., 0] * pi[..., 1]
                        + rates[..., 1] * pi[..., 2]
                        + rates[..., 2] * pi[..., 3]
                    ),
                    rates[..., 0] * pi[..., 1],
                    rates[..., 1] * pi[..., 2],
                    rates[..., 2] * pi[..., 3],
                ],
                [
                    rates[..., 0] * pi[..., 0],
                    -(
                        rates[..., 0] * pi[..., 0]
                        + rates[..., 3] * pi[..., 2]
                        + rates[..., 4] * pi[..., 3]
                    ),
                    rates[..., 3] * pi[..., 2],
                    rates[..., 4] * pi[..., 3],
                ],
                [
                    rates[..., 1] * pi[..., 0],
                    rates[..., 3] * pi[..., 1],
                    -(
                        rates[..., 1] * pi[..., 0]
                        + rates[..., 3] * pi[..., 1]
                        + rates[..., 5] * pi[..., 3]
                    ),
                    rates[..., 5] * pi[..., 3],
                ],
                [
                    rates[..., 2] * pi[..., 0],
                    rates[..., 4] * pi[..., 1],
                    rates[..., 5] * pi[..., 2],
                    -(
                        rates[..., 2] * pi[..., 0]
                        + rates[..., 4] * pi[..., 1]
                        + rates[..., 5] * pi[..., 2]
                    ),
                ],
            ],
        )


__all__ = ["GTR"]
