import tensorflow as tf
import tensorflow_probability as tfp

COALESCENCE, SAMPLING, OTHER = -1, 1, 0

def coalescent_likelihood(lineage_count,
                          population_func, # At coalescence
                          population_areas, # Integrals of 1/N
                          coalescent_mask): # At top of interval
    k_choose_2 = lineage_count * (lineage_count - 1) / 2
    return -tf.reduce_sum(k_choose_2 * population_areas) - tf.reduce_sum(tf.math.log(tf.boolean_mask(population_func, coalescent_mask)))

def get_lineage_count(event_types):
    return tf.math.cumsum(event_types)

class ConstantCoalescent(tfp.distributions.Distribution):
    def __init__(self, pop_size,node_mask,
               validate_args=False,
               allow_nan_stats=True,
               name='ConstantCoalescent'):
        super(ConstantCoalescent, self).__init__(
            dtype=None,
            reparameterization_type=tfp.distributions.NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters = dict(locals()),
            name=name
        ) # TODO: Values for other args
        self.pop_size = pop_size
        self.node_mask = node_mask

    def _log_prob(self, heights):
        sort_indices = tf.argsort(heights)
        heights_sorted = tf.gather(heights, sort_indices)
        node_mask_sorted = tf.gather(self.node_mask, sort_indices)

        lineage_count =  get_lineage_count(tf.where(node_mask_sorted, COALESCENCE, SAMPLING))[:-1]
        population_func = tf.broadcast_to(tf.expand_dims(self.pop_size, 0), lineage_count.shape) # To broadcast
        durations = heights_sorted[1:] - heights_sorted[:-1]
        population_areas = durations / self.pop_size
        coalescent_mask = node_mask_sorted[1:]

        return coalescent_likelihood(lineage_count, population_func, population_areas, coalescent_mask)