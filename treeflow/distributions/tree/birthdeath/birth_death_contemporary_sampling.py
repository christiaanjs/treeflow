import typing as tp

from treeflow.distributions.tree.rooted_tree_distribution import RootedTreeDistribution
import tensorflow as tf
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.util import ParameterProperties
from tensorflow_probability.python.bijectors.softplus import (
    Softplus as SoftplusBijector,
)
from tensorflow_probability.python.bijectors.sigmoid import Sigmoid as SigmoidBijector
from tensorflow_probability.python.internal import dtype_util
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree


class BirthDeathContemporarySampling(RootedTreeDistribution):
    """
    Stadler, T. (2009). On incomplete sampling under birth–death models and connections
    to the sampling-based coalescent. Journal of theoretical biology, 261(1), 58-66.

    Parameters
    ----------
    birth_diff_rate
        Birth difference rate parameter, lambda - mu in birth/death model (diversification rate)
    relative_death_rate
        Relative death rate parameter, mu/lambda in birth death model (turnover parameter)
    sample_probability
        Sample probability, rho in birth/death model
    """

    def __init__(
        self,
        taxon_count,
        birth_diff_rate,
        relative_death_rate,
        sample_probability=1.0,
        validate_args=False,
        allow_nan_stats=True,
        name="BirthDeathContemporarySampling",
        tree_name: tp.Optional[str] = None,
    ):
        self.birth_diff_rate = tensor_util.convert_nonref_to_tensor(birth_diff_rate)
        dtype = self.birth_diff_rate.dtype
        self.relative_death_rate = tensor_util.convert_nonref_to_tensor(
            relative_death_rate
        )
        self.sample_probability = tensor_util.convert_nonref_to_tensor(
            sample_probability, dtype=dtype
        )
        super().__init__(
            taxon_count,
            sampling_time_reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            node_height_reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            time_dtype=dtype,
            parameters=dict(locals()),
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            tree_name=tree_name,
        )

    def _sample_n(self, n, seed=None):
        import warnings

        warnings.warn("Dummy sampling")
        dtype = self.dtype
        event_shape = self.event_shape_tensor()

        shape_func = lambda event_shape: tf.concat([[n], event_shape], axis=0)

        return tf.nest.map_structure(
            lambda event_shape, dtype: tf.zeros(shape_func(event_shape), dtype),
            event_shape,
            dtype,
        )

    def _log_prob(self, x: TensorflowRootedTree):
        heights: tf.Tensor = x.heights[self.taxon_count :]
        dtype = heights.dtype
        taxon_count = tf.cast(self.taxon_count, dtype)
        r = self.birth_diff_rate
        a = self.relative_death_rate
        rho = self.sample_probability

        log_coeff = (taxon_count - 1) * tf.math.log(
            tf.constant(2.0, dtype=dtype)
        ) - tf.math.lgamma(taxon_count)
        tree_logp = (
            log_coeff
            + (taxon_count - 1) * tf.math.log(r * rho)
            + taxon_count * tf.math.log(1 - a)
        )

        mrhs = tf.expand_dims(-r, -1) * heights

        zs = tf.math.log(
            tf.expand_dims(rho, -1)
            + tf.expand_dims((1 - rho) - a, -1) * tf.math.exp(mrhs)
        )
        ls = -2 * zs + mrhs
        root_term = mrhs[..., -1] - zs[..., -1]

        return tree_logp + tf.reduce_sum(ls, axis=-1) + root_term

    @classmethod
    def _parameter_properties(
        cls, dtype, num_classes=None
    ) -> tp.Dict[str, ParameterProperties]:
        return dict(
            birth_diff_rate=ParameterProperties(
                event_ndims=0,
                default_constraining_bijector_fn=(
                    lambda: SoftplusBijector(low=dtype_util.eps(dtype))
                ),
            ),
            relative_death_rate=ParameterProperties(
                event_ndims=0,
                default_constraining_bijector_fn=(lambda: SigmoidBijector()),
            ),
            sample_probability=ParameterProperties(
                event_ndims=0,
                default_constraining_bijector_fn=lambda: SigmoidBijector(),
            ),
        )
