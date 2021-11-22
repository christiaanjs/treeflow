from abc import abstractmethod
import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF


class BaseSiteModel:
    def __init__(self, category_count, normalise_rates=True):
        self.category_count = category_count
        self.normalise_rates = normalise_rates

    def weights(self, dtype=DEFAULT_FLOAT_DTYPE_TF):
        return tf.fill(
            [self.category_count], tf.constant(1.0, dtype=dtype) / self.category_count
        )

    def p(self, dtype=DEFAULT_FLOAT_DTYPE_TF):
        return (2 * tf.range(self.category_count, dtype=DEFAULT_FLOAT_DTYPE_TF) + 1) / (
            2 * self.category_count
        )

    @abstractmethod
    def quantile(self, p, **params):
        ...

    def rates_weights(self, **params):
        weights = self.weights()
        p = self.p()
        rates = self.quantile(p, **params)
        return (
            rates / tf.reduce_sum(rates * weights) if self.normalise_rates else rates,
            weights,
        )
