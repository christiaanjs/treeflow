import tensorflow as tf
from treeflow.evolution.substitution.eigendecomposition import Eigendecomposition
from treeflow.evolution.substitution.base_substitution_model import (
    EigendecompositionSubstitutionModel,
)
from treeflow.evolution.substitution.nucleotide.alphabet import A, C, G, T


def pack_matrix(mat):
    return tf.stack([tf.stack(row, axis=-1) for row in mat], axis=-2)


def pack_matrix_transposed(mat):
    return tf.stack([tf.stack(col, axis=-1) for col in mat], axis=-1)


class HKY(EigendecompositionSubstitutionModel):
    def q(self, frequencies: tf.Tensor, kappa: tf.Tensor) -> tf.Tensor:
        pi = frequencies

        piA = pi[..., A]
        piC = pi[..., C]
        piT = pi[..., T]
        piG = pi[..., G]
        return pack_matrix(
            [
                [-(piC + kappa * piG + piT), piC, kappa * piG, piT],
                [piA, -(piA + piG + kappa * piT), piG, kappa * piT],
                [kappa * piA, piC, -(kappa * piA + piC + piT), piT],
                [piA, kappa * piC, piG, -(piA + kappa * piC + piG)],
            ],
        )

    def eigen(self, frequencies: tf.Tensor, kappa: tf.Tensor) -> Eigendecomposition:
        pi = frequencies
        piA = pi[..., A]
        piC = pi[..., C]
        piT = pi[..., T]
        piG = pi[..., G]
        piY = piT + piC
        piR = piA + piG

        beta = -1.0 / (2.0 * (piR * piY + kappa * (piA * piG + piC * piT)))

        one = tf.ones_like(kappa)
        zero = tf.zeros_like(kappa)
        minus_one = -one

        eigenvalues = tf.stack(
            [
                zero,
                beta,
                beta * (piY * kappa + piR),
                beta * (piY + piR * kappa),
            ],
            axis=-1,
        )
        eigenvectors = pack_matrix_transposed(
            [
                [one, one, one, one],
                [1.0 / piR, -1.0 / piY, 1.0 / piR, -1.0 / piY],
                [zero, piT / piY, zero, -piC / piY],
                [piG / piR, zero, -piA / piR, zero],
            ],
        )
        inverse_eigenvectors = pack_matrix(
            [
                [piA, piC, piG, piT],
                [piA * piY, -piC * piR, piG * piY, -piT * piR],
                [zero, one, zero, minus_one],
                [one, zero, minus_one, zero],
            ],
        )

        return Eigendecomposition(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            inverse_eigenvectors=inverse_eigenvectors,
        )


__all__ = [HKY.__name__]
