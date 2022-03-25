from typing_extensions import Protocol
import tensorflow as tf


class FiniteDiscreteDistribution(Protocol):
    @property
    def support(self) -> tf.Tensor:
        """Values that discrete distribution can take"""
        ...

    @property
    def probabilities(self):
        """Respective probability masses for support"""
        ...

    @property
    def support_size(self):
        """Number of elements in support"""
        ...
