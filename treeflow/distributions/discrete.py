from typing_extensions import Protocol
import tensorflow as tf


class FiniteDiscreteDistribution(Protocol):
    """Interface for discrete probability distributions with a finite support"""

    @property
    def support(self) -> tf.Tensor:
        """Values that discrete distribution can take"""
        ...

    @property
    def normalised_support(self) -> tf.Tensor:
        """Support normalised to have a mean of 1 (weighted by probability mass)"""
        ...

    @property
    def probabilities(self):
        """Respective probability masses for support"""
        ...

    @property
    def support_size(self):
        """Number of elements in support"""
        ...
