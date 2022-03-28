import tensorflow as tf
import typing as tp
from tensorflow_probability.python.distributions import (
    MixtureSameFamily,
    Distribution,
    Categorical,
)
from treeflow.distributions.discrete import FiniteDiscreteDistribution
from tensorflow_probability.python.internal import parameter_properties


class DiscreteParameterMixture(MixtureSameFamily):
    """
    Class to represent a distribution which depends on a random
    discrete parameter that is summed out
    """

    def __init__(
        self,
        discrete_distribution: FiniteDiscreteDistribution,
        components_distribution: Distribution,
        reparameterize=False,
        validate_args=False,
        name=None,
    ):
        """
        Parameters
        ----------
        discrete_distribution
        components_distribution
            Must have batch dim for discrete parameter
        reparameterize
            Whether to reparameterize samples of the distribution using implicit
            reparameterization gradients
        validate_args
        name
        """
        parameters = locals()
        if name is None:
            name = "Marginalized" + components_distribution.name
        self._discrete_distribution = discrete_distribution
        self._components_distribution = components_distribution
        mixture_distribution = Categorical(
            probs=self._discrete_distribution.probabilities
        )
        super().__init__(
            mixture_distribution=mixture_distribution,
            components_distribution=components_distribution,
            reparameterize=reparameterize,
            validate_args=validate_args,
            allow_nan_stats=components_distribution.allow_nan_stats,
            name=name,
        )
        self._parameters = parameters

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            discrete_distribution=(parameter_properties.BatchedComponentProperties()),
            components_distribution=(
                parameter_properties.BatchedComponentProperties(event_ndims=1)
            ),
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
