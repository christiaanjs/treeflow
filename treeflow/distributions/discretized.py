import tensorflow as tf
from tensorflow_probability.python.distributions import (
    Distribution,
    Categorical,
    NOT_REPARAMETERIZED,
)


class DiscretizedDistribution(Distribution):
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

        self._quantiles = self._distribution.quantile(self._probabilities)
        self._mass = 1.0 / self._category_count_float
        self._index_dist = Categorical(probs=tf.fill(self._category_count, self._mass))

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
        indices = tf.random.uniform(
            tf.expand_dims(n, 0), maxval=self._category_count, seed=seed, dtype=tf.int32
        )
        return tf.gather(self._quantiles, indices, axis=-1)

    @property
    def quantiles(self):
        return self._quantiles

    @property
    def probabilities(self):
        return tf.fill(self._category_count, self._mass)
