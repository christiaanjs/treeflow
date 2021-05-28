import tensorflow as tf
import tensorflow_probability as tfp
import treeflow.tf_util
import treeflow.tree
from treeflow import DEFAULT_FLOAT_DTYPE_TF

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
def coalescent_likelihood(
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


class ConstantCoalescent(treeflow.tree.TreeDistribution):
    def __init__(
        self,
        taxon_count,
        pop_size,
        sampling_times,
        validate_args=False,
        allow_nan_stats=True,
        name="ConstantCoalescent",
    ):
        super(ConstantCoalescent, self).__init__(
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=dict(locals()),
            name=name,
        )
        self.pop_size = pop_size
        self.sampling_times = sampling_times
        self.taxon_count = taxon_count  # tf.convert_to_tensor(taxon_count)

    def _log_prob_1d(self, x, pop_size_1d):
        # TODO: Validate topology
        # TODO: Check sampling times?
        heights = x["heights"]
        node_mask = tf.concat(
            [tf.fill([self.taxon_count], False), tf.fill([self.taxon_count - 1], True)],
            0,
        )

        sort_indices = tf.argsort(heights)
        heights_sorted = tf.gather(heights, sort_indices)
        node_mask_sorted = tf.gather(node_mask, sort_indices)

        lineage_count = get_lineage_count(
            tf.where(node_mask_sorted, COALESCENCE, SAMPLING)
        )[:-1]
        population_func = tf.broadcast_to(pop_size_1d, tf.shape(lineage_count))
        durations = heights_sorted[1:] - heights_sorted[:-1]
        population_areas = durations / pop_size_1d
        coalescent_mask = node_mask_sorted[1:]

        return coalescent_likelihood(
            lineage_count, population_func, population_areas, coalescent_mask
        )

    def _log_prob_1d_flat(self, x_flat):
        x_dict = {"heights": x_flat[0], "topology": {"parent_indices": x_flat[1]}}
        pop_size = x_flat[2]
        return self._log_prob_1d(x_dict, pop_size)

    def _log_prob(self, x):
        batch_shape = tf.shape(x["heights"])[:-1]
        pop_size = tf.broadcast_to(self.pop_size, batch_shape)
        x_flat = [x["heights"], x["topology"]["parent_indices"], pop_size]
        return treeflow.tf_util.vectorize_1d_if_needed(
            self._log_prob_1d_flat, x_flat, tf.shape(batch_shape)[0]
        )

    def _sample_n(self, n, seed=None):
        import warnings

        warnings.warn("Dummy sampling")
        # raise NotImplementedError('Coalescent simulator not yet implemented')
        return {
            "heights": tf.zeros(
                [n, 2 * self.taxon_count - 1], dtype=self.dtype["heights"]
            ),
            "topology": {
                "parent_indices": tf.zeros(
                    [n, 2 * self.taxon_count - 2],
                    dtype=self.dtype["topology"]["parent_indices"],
                )
            },
        }
