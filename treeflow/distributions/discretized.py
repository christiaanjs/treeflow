import tensorflow as tf
from tensorflow_probability.python.distributions import (
    Distribution,
    Categorical,
    NOT_REPARAMETERIZED,
)
from tensorflow_probability.python.internal import parameter_properties


class DiscretizedDistribution(Distribution):
    """A discrete distribution created by"""

    def __init__(
        self,
        category_count: tf.Tensor,
        distribution: Distribution,
        epsilon=1e-16,
        validate_args=False,
        name=None,
    ):
        parameters = locals()
        if name is None:
            name = "Discretized" + distribution.name
        self._category_count = category_count
        self._distribution = distribution
        self._epsilon = tf.convert_to_tensor(epsilon, dtype=self._distribution.dtype)

        float_indices = tf.cast(
            tf.range(self._category_count), self._distribution.dtype
        )
        self._category_count_float = tf.cast(
            self._category_count, self._distribution.dtype
        )
        self._probabilities = (2.0 * float_indices + 1.0) / (
            2.0 * self._category_count_float
        )
        batch_shape = self._distribution.batch_shape_tensor()
        self._probabilities_b = tf.reshape(
            self._probabilities,
            tf.concat(
                [
                    [category_count],
                    tf.ones_like(batch_shape),
                ],
                axis=0,
            ),
        )

        batch_rank = tf.shape(batch_shape)[0]
        self._quantiles = tf.transpose(
            self._distribution.quantile(self._probabilities_b),
            tf.concat([tf.range(1, batch_rank + 1), [0]], axis=0),
        )
        self._mass = 1.0 / self._category_count_float
        self._mass_b = tf.broadcast_to(
            self._mass,
            tf.concat(
                [
                    batch_shape,
                    [category_count],
                ],
                axis=0,
            ),
        )  # Category on last axis
        probs_shape = tf.reshape(self._category_count, (1,))
        self._index_dist = Categorical(probs=tf.fill(probs_shape, self._mass))

        super().__init__(
            dtype=self._distribution.dtype,
            reparameterization_type=NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=distribution.allow_nan_stats,
            parameters=parameters,
            name=name,
        )

    def _prob(self, value):
        close = tf.abs(tf.expand_dims(value, -1) - self._quantiles) < self._epsilon
        supported = tf.reduce_any(close, axis=-1)
        return tf.where(supported, self._mass, 0.0)

    def _sample_n(self, n, seed=None):
        indices = Categorical(probs=self._mass_b).sample(n)
        indices_b = tf.expand_dims(indices, -1)
        quantiles_b = tf.broadcast_to(
            self._quantiles,
            tf.concat([tf.expand_dims(n, 0), tf.shape(self._quantiles)], axis=0),
        )
        return tf.gather(quantiles_b, indices, axis=-1, batch_dims=-1)[..., 0]

    @property
    def support_size(self):
        return self._category_count

    @property
    def support(self):
        return self._quantiles

    @property
    def normalised_support(self):
        support = self.support
        return support / tf.reduce_mean(support, axis=-1, keepdims=True)

    @property
    def probabilities(self):
        return tf.fill(tf.expand_dims(self._category_count, 0), self._mass)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            distribution=(parameter_properties.BatchedComponentProperties()),
        )


__all__ = ["DiscretizedDistribution"]
