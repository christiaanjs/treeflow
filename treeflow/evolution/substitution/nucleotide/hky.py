import tensorflow as tf
from treeflow.evolution.substitution.eigendecomposition import Eigendecomposition
from treeflow.evolution.substitution.base_substitution_model import (
    EigendecompositionSubstitutionModel,
)
from treeflow.evolution.substitution.nucleotide.alphabet import A, C, G, T


def pack_matrix(mat):
    return tf.concat([tf.concat(row, axis=-1) for row in mat], axis=-2)


def pack_matrix_transposed(mat):
    return tf.concat([tf.concat(col, axis=-1) for col in mat], axis=-1)


class HKY(EigendecompositionSubstitutionModel):
    def q(self, frequencies: tf.Tensor, kappa: tf.Tensor) -> tf.Tensor:
        pass

    def eigen(self, frequencies: tf.Tensor, kappa: tf.Tensor) -> Eigendecomposition:
        pi = frequencies
        piA = pi[..., A]
        piC = pi[..., C]
        piT = pi[..., T]
        piG = pi[..., G]
        piY = piT + piC
        piR = piA + piG

        beta = -1.0 / (2.0 * (piR * piY + kappa * (piA * piG + piC * piT)))

        batch_shape = tf.shape(kappa)  # TODO: Should this use prefer_static?
        one = tf.broadcast_to(1.0, batch_shape)
        zero = tf.broadcast_to(0.0, batch_shape)
        minus_one = -one

        eigenvalues = tf.concat(
            [
                0.0,
                beta,
                beta * (piY * kappa + piR),
                beta * (piY + piR * kappa),
            ],
            axis=-1,
        )
        eigenvectors = pack_matrix(
            [
                [one, one, one, one],
                [1.0 / piR, -1.0 / piY, 1.0 / piR, -1.0 / piY],
                [zero, piT / piY, zero, -piC / piY],
                [piG / piR, zero, -piA / piR, zero],
            ]
        )
        inverse_eigenvectors = pack_matrix_transposed(
            [
                [piA, piC, piG, piT],
                [piA * piY, -piC * piR, pi[G] * piY, -piT * piR],
                [zero, one, zero, minus_one],
                [one, zero, minus_one, zero],
            ]
        )

        return Eigendecomposition(
            eigenvalues,
            eigenvectors=eigenvectors,
            inverse_eigenvectors=inverse_eigenvectors,
        )


__all__ = [HKY.__name__]
