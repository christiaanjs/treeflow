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
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.traversal.anchor_heights import get_anchor_heights_tensor


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
        fixed_topology=None,
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
        self.fixed_topology = fixed_topology
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
            support_topology_batch_dims=True,
        )

    def _sample_n(self, n, seed=None):
        from treeflow.distributions.tree.birthdeath.cpp_sampler import sample_bd_tree

        # n and taxon_count must be Python ints for the pure-TF sampler to
        # build static-shape tensors.  During tf.function tracing they are
        # usually static Python ints; fall back to dummy samples only when
        # they are genuinely symbolic (e.g. TFP JointDistribution get_output_spec).
        try:
            n_int = int(n)
            n_taxa_int = int(self.taxon_count)
        except (TypeError, ValueError):
            return self._make_dummy_samples(
                tf.zeros(self.taxon_count, dtype=self.dtype.sampling_times), n
            )

        r = self.birth_diff_rate
        a = self.relative_death_rate
        # Convert (r = lambda-mu, a = mu/lambda) → (lambda, mu)
        lambda_ = r / (tf.cast(1.0, r.dtype) - a)
        mu = r * a / (tf.cast(1.0, r.dtype) - a)
        rho = self.sample_probability

        return sample_bd_tree(
            n_taxa=n_taxa_int,
            lambda_=lambda_,
            mu=mu,
            rho=rho,
            n_samples=n_int,
            seed=seed,
            fixed_topology=self.fixed_topology,
        )

    def _log_coeff(self, dtype):
        taxon_count = tf.cast(self.taxon_count, dtype)
        return (taxon_count - 1) * tf.math.log(
            tf.constant(2.0, dtype=dtype)
        ) - tf.math.lgamma(taxon_count)

    def _log_prob(self, x: TensorflowRootedTree):
        heights: tf.Tensor = x.node_heights
        dtype = heights.dtype
        taxon_count = tf.cast(self.taxon_count, dtype)
        r = self.birth_diff_rate
        a = self.relative_death_rate
        rho = self.sample_probability

        log_coeff = self._log_coeff(dtype)
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

    @property
    def sampling_times(self):
        return tf.zeros(self.taxon_count, dtype=self.dtype.node_heights)

    def _default_event_space_bijector(self, topology: TensorflowTreeTopology):
        from treeflow.bijectors.tree_ratio_bijector import TreeRatioBijector

        anchor_heights = tf.zeros(self.taxon_count - 1, dtype=self.dtype.node_heights)
        return TreeRatioBijector(
            topology=topology,
            anchor_heights=anchor_heights,
            fixed_sampling_times=True,
        )


__all__ = ["BirthDeathContemporarySampling"]
