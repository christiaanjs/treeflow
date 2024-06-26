from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
import typing as tp
import tensorflow as tf
import tensorflow_probability as tfp
from treeflow.tf_util.vectorize import broadcast_structure, vectorize_over_batch_dims
from treeflow.distributions.tree.rooted_tree_distribution import RootedTreeDistribution
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.util import ParameterProperties
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.internal import dtype_util
from treeflow.traversal.anchor_heights import get_anchor_heights_tensor

COALESCENCE, SAMPLING, OTHER = -1, 1, 0


def get_lineage_count(event_types):
    return tf.math.cumsum(event_types)


# tf.boolean_mask is picky about mask dimension; required to get this working in function mode
@tf.function(
    input_signature=[
        tf.TensorSpec(None, tf.int32),
        tf.TensorSpec(None, DEFAULT_FLOAT_DTYPE_TF),
        tf.TensorSpec(None, DEFAULT_FLOAT_DTYPE_TF),
        tf.TensorSpec([None], tf.bool),
    ]
)
def base_coalescent_log_likelihood(
    lineage_count,
    population_func,  # At coalescence
    population_areas,  # Integrals of 1/N
    coalescent_mask,
):  # At top of interval
    k_choose_2 = (
        tf.cast(lineage_count * (lineage_count - 1), dtype=DEFAULT_FLOAT_DTYPE_TF) / 2.0
    )
    return -tf.reduce_sum(k_choose_2 * population_areas) - tf.reduce_sum(
        tf.math.log(tf.boolean_mask(population_func, coalescent_mask))
    )


def _constant_coalescent_log_likelihood(
    args: tp.Tuple[TensorflowRootedTree, tf.Tensor]
):
    tree, pop_size = args
    heights = tree.heights
    taxon_count = tree.taxon_count
    node_mask = tf.concat(
        [tf.fill([taxon_count], False), tf.fill([taxon_count - 1], True)],
        0,
    )
    sort_indices = tf.argsort(heights)
    heights_sorted = tf.gather(heights, sort_indices)
    node_mask_sorted = tf.gather(node_mask, sort_indices)

    lineage_count = get_lineage_count(
        tf.where(node_mask_sorted, COALESCENCE, SAMPLING)
    )[:-1]
    population_func = tf.broadcast_to(pop_size, tf.shape(lineage_count))
    durations = heights_sorted[1:] - heights_sorted[:-1]
    population_areas = durations / pop_size
    coalescent_mask = node_mask_sorted[1:]

    return base_coalescent_log_likelihood(
        lineage_count, population_func, population_areas, coalescent_mask
    )


class ConstantCoalescent(RootedTreeDistribution):
    def __init__(
        self,
        taxon_count: int,
        pop_size: tf.Tensor,
        sampling_times: tf.Tensor,
        validate_args=False,
        allow_nan_stats=True,
        name="ConstantCoalescent",
        tree_name: tp.Optional[str] = None,
    ):
        """
        Tree prior distribution based on a coalescent model with a constant population size and
        serial sampling.

        Parameters
        ----------
        taxon_count : int
            Number of leaf taxa in the tree
        pop_size : Tensor
            Effective population size parameter (scalar, but can be vectorized over)
        sampling_times : Tensor
            Tensor of sampling times of size `taxon_count`
        tree_name : str
            Internal name assigned to this tree, separate to distribution name (can be used by TreeFlow)
        """
        super().__init__(
            taxon_count=taxon_count,
            node_height_reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            sampling_time_reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            time_dtype=sampling_times.dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=dict(locals()),
            name=name,
            tree_name=tree_name,
        )
        self.pop_size = tensor_util.convert_nonref_to_tensor(pop_size)
        self.sampling_times = tensor_util.convert_nonref_to_tensor(sampling_times)

    def _log_prob(self, x: TensorflowRootedTree):
        event_ndims = self.event_shape.node_heights.rank
        batch_shape = tf.broadcast_dynamic_shape(
            self.batch_shape_tensor(), tf.shape(x.node_heights)[:-event_ndims]
        )
        pop_size = tf.broadcast_to(self.pop_size, batch_shape)
        tree_shape = self.event_shape_tensor()
        tree = broadcast_structure(x, tree_shape, batch_shape)
        return vectorize_over_batch_dims(
            _constant_coalescent_log_likelihood,
            (tree, pop_size),
            (tree_shape, tf.convert_to_tensor((), dtype=tf.int32)),
            batch_shape,
            vectorized_map=False,
            fn_output_signature=DEFAULT_FLOAT_DTYPE_TF,  # Argsort doesn't work with pfor
        )

    def _sample_n(self, n, seed=None):
        import warnings

        warnings.warn("Dummy sampling")
        return self._make_dummy_samples(self.sampling_times, n)

    @classmethod
    def _parameter_properties(
        cls, dtype, num_classes=None
    ) -> tp.Dict[str, ParameterProperties]:
        return dict(
            pop_size=ParameterProperties(
                event_ndims=0,
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))
                ),
            ),
            sampling_times=ParameterProperties(
                event_ndims=1,
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))
                ),
            ),
        )

    def _default_event_space_bijector(self, topology: TensorflowTreeTopology):
        from treeflow.bijectors.tree_ratio_bijector import TreeRatioBijector

        return TreeRatioBijector(
            topology=topology,
            anchor_heights=get_anchor_heights_tensor(topology, self.sampling_times),
            fixed_sampling_times=True,
        )


__all__ = ["ConstantCoalescent"]
