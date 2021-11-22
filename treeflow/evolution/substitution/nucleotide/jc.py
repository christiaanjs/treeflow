import typing as tp
from treeflow import DEFAULT_FLOAT_DTYPE_TF
import tensorflow as tf
import numpy as np
from treeflow.evolution.substitution.eigendecomposition import Eigendecomposition
from treeflow.evolution.substitution.base_substitution_model import (
    EigendecompositionSubstitutionModel,
)


class JC(EigendecompositionSubstitutionModel):
    @staticmethod
    def frequencies(dtype=DEFAULT_FLOAT_DTYPE_TF):
        return tf.constant([1 / 4] * 4, dtype=dtype)

    def q(self, frequencies: tf.Tensor, dtype=DEFAULT_FLOAT_DTYPE_TF) -> tf.Tensor:
        return tf.constant(
            [
                [-1, 1 / 3, 1 / 3, 1 / 3],
                [1 / 3, -1, 1 / 3, 1 / 3],
                [1 / 3, 1 / 3, -1, 1 / 3],
                [1 / 3, 1 / 3, 1 / 3, -1],
            ],
            dtype=dtype,
        )

    def eigen(
        self,
        frequencies: tp.Optional[tf.Tensor] = None,
        dtype: tf.DType = DEFAULT_FLOAT_DTYPE_TF,
    ) -> Eigendecomposition:
        return Eigendecomposition(
            eigenvectors=tf.constant(
                [
                    [1.0, 2.0, 0.0, 0.5],
                    [1.0, -2.0, 0.5, 0.0],
                    [1.0, 2.0, 0.0, -0.5],
                    [1.0, -2.0, -0.5, 0.0],
                ],
                dtype=dtype,
            ),
            eigenvalues=tf.constant(
                [0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333],
                dtype=dtype,
            ),
            inverse_eigenvectors=tf.constant(
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.125, -0.125, 0.125, -0.125],
                    [0.0, 1.0, 0.0, -1.0],
                    [1.0, 0.0, -1.0, 0.0],
                ],
                dtype=dtype,
            ),
        )


__all__ = [JC.__name__]
