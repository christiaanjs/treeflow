import tensorflow as tf
from abc import abstractmethod
from treeflow.evolution.substitution.eigendecomposition import (
    Eigendecomposition,
)


def normalising_constant(q: tf.Tensor, pi: tf.Tensor) -> tf.Tensor:
    return -tf.reduce_sum(tf.linalg.diag_part(q) * pi)


def normalise(q: tf.Tensor, pi: tf.Tensor) -> tf.Tensor:
    return q / normalising_constant(q, pi)


class SubstitutionModel:
    @abstractmethod
    def q(self, frequencies: tf.Tensor, **kwargs: tf.Tensor) -> tf.Tensor:
        ...

    def q_norm(self, frequencies: tf.Tensor, **kwargs: tf.Tensor) -> tf.Tensor:
        return normalise(self.q(frequencies, **kwargs), frequencies)


class EigendecompositionSubstitutionModel(
    SubstitutionModel
):  # TODO: Rename class to time reversible, method to diagonalisation?
    def eigen(self, frequencies: tf.Tensor, **kwargs: tf.Tensor) -> Eigendecomposition:
        """Eigendecomposition of the normalised instantaneous rate matrix"""

        # First create a symmetric matrix from the normalised Q matrix
        q_norm = self.q_norm(frequencies=frequencies, **kwargs)
        sqrt_frequencies = tf.math.sqrt(frequencies)
        inverse_sqrt_frequencies = 1.0 / sqrt_frequencies

        sqrt_frequencies_diag_matrix = tf.linalg.diag(sqrt_frequencies)
        inverse_sqrt_frequencies_diag_matrix = tf.linalg.diag(inverse_sqrt_frequencies)

        symmetric_matrix = (
            sqrt_frequencies_diag_matrix @ q_norm @ inverse_sqrt_frequencies_diag_matrix
        )
        eigenvalues, s_eigenvectors = tf.linalg.eigh(symmetric_matrix)
        eigenvectors = inverse_sqrt_frequencies_diag_matrix @ s_eigenvectors
        inverse_eigenvectors = (
            tf.linalg.matrix_transpose(s_eigenvectors) @ sqrt_frequencies_diag_matrix
        )

        return Eigendecomposition(
            eigenvectors=eigenvectors,
            inverse_eigenvectors=inverse_eigenvectors,
            eigenvalues=eigenvalues,
        )


__all__ = [SubstitutionModel.__name__, EigendecompositionSubstitutionModel.__name__]
