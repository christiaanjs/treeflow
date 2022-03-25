import tensorflow as tf
import typing as tp
from tensorflow_probability.python.distributions import (
    MixtureSameFamily,
    Distribution,
    Categorical,
)
from treeflow.distributions.discrete import FiniteDiscreteDistribution


class DiscreteParameterMixture(MixtureSameFamily):
    """
    Class to represent a distribution which depends on a random
    discrete parameter that is summed out
    """

    def __init__(
        self,
        discrete_distribution: FiniteDiscreteDistribution,
        dist_function: tp.Callable[
            [object], Distribution
        ],  # TODO: Allow kwargs in dist_function
        reparameterize=False,
        validate_args=False,
        name=None,
    ):
        """
        Parameters
        ----------
        discrete_distribution
        dist_function
            Must vectorise over discrete parameter
        reparameterize
            Whether to reparameterize samples of the distribution using implicit
            reparameterization gradients
        validate_args
        name
        """
        params = locals()
        concrete = dist_function(discrete_distribution.sample())
        if name is None:
            name = "Marginalised" + concrete.name
        self._discrete_distribution = discrete_distribution
        self._dist_function = dist_function
        mixture_distribution = Categorical(
            probs=self._discrete_distribution.probabilities
        )
        components_distribution = self._dist_function(
            self._discrete_distribution.support
        )
        super().__init__(
            mixture_distribution=mixture_distribution,
            components_distribution=components_distribution,
            validate_args=validate_args,
            allow_nan_stats=concrete.allow_nan_stats,
            name=name,
        )

    # def _sample_n(self, n, seed=None) -> tf.Tensor:
    #     discrete_samples = self._discrete_distribution.sample(n, seed=seed)
    #     concrete = self._dist_function(discrete_samples)
    #     return concrete.sample(seed=seed)  # Sample shape already part of batch shape

    # def _prob(self, x) -> tf.Tensor:  # TODO: Support nesting
    #     # TODO
    #     event_ndims = tf.shape(concrete.event_shape)[0]
    #     x_b = tf.expand_dims(x, -event_ndims)
    #     probs = concrete.prob(x_b)
    #     weights = self._discrete_distribution.probabilities
    #     weights_shape = tf.concat(
    #         [tf.shape(weights), tf.ones(event_ndims, dtype=tf.int32)],
    #         axis=0,
    #     )
    #     weights_b = tf.reshape(weights, weights_shape)
    #     summed = tf.reduce_sum(weights_b * probs, axis=-event_ndims)
    #     return summed
