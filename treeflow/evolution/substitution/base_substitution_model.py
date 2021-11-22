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


class EigendecompositionSubstitutionModel(SubstitutionModel):
    @abstractmethod
    def eigen(self, frequencies: tf.Tensor, **kwargs: tf.Tensor) -> Eigendecomposition:
        """Eigendecomposition of the normalised instantaneous rate matrix"""


__all__ = [SubstitutionModel.__name__, EigendecompositionSubstitutionModel.__name__]
