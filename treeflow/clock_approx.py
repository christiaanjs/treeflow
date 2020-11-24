
import tensorflow as tf
import tensorflow_probability as tfp
import treeflow.sequences

def get_tree_statistic(tree, tree_statistic):
    if tree_statistic == "length":
        blens = treeflow.sequences.get_branch_lengths(tree)
        return tf.reduce_sum(blens, axis=-1)
    elif tree_statistic == "height":
        return tree["heights"][..., -1]
    else:
        raise ValueError(f"Unknown tree statistic {tree_statistic}")

class ScaledDistribution(tfp.distributions.TransformedDistribution):
    def __init__(self, distribution, tree, tree_statistic="length", **dist_kwargs):
        statistic = get_tree_statistic(tree, tree_statistic)
        batch_shape = tf.shape(statistic)
        bij = tfp.bijectors.Scale(1.0 / statistic)

        if callable(distribution):
            dist_kwargs_b = {}
            for key, val in dist_kwargs.items():
                arg_shape = tf.concat([batch_shape, tf.shape(val)], 0)
                dist_kwargs_b[key] = tf.broadcast_to(val, arg_shape)
            distribution = distribution(**dist_kwargs_b)

        super(ScaledDistribution, self).__init__(distribution=distribution, bijector=bij)

class ScaledRateDistribution(tfp.distributions.TransformedDistribution):
    def __init__(self, distribution, tree, clock_rate, **dist_kwargs): # TODO: dist_kwargs_event_ndims for different kwrags
        blens = treeflow.sequences.get_branch_lengths(tree)
        batch_shape = tf.shape(clock_rate)
        bij = tfp.bijectors.Scale(1.0 / (blens * tf.expand_dims(clock_rate, -1)))

        if callable(distribution):
            dist_kwargs_b = {}
            for key, val in dist_kwargs.items():
                arg_shape = tf.concat([batch_shape, tf.shape(val)], 0)
                dist_kwargs_b[key] = tf.broadcast_to(val, arg_shape)
            distribution = distribution(**dist_kwargs_b)
        
        super(ScaledRateDistribution, self).__init__(distribution=distribution, bijector=bij)

class TuneableScaledRateDistribution(tfp.distributions.TransformedDistribution):
    def __init__(self, distribution, scale_power, tree, clock_rate, **dist_kwargs): # TODO: dist_kwargs_event_ndims for different kwrags
        blens = treeflow.sequences.get_branch_lengths(tree)
        batch_shape = tf.shape(clock_rate)
        bij = tfp.bijectors.Scale(1.0 / tf.math.pow(blens * tf.expand_dims(clock_rate, -1), scale_power))

        if callable(distribution):
            dist_kwargs_b = {}
            for key, val in dist_kwargs.items():
                arg_shape = tf.concat([batch_shape, tf.shape(val)], 0)
                dist_kwargs_b[key] = tf.broadcast_to(val, arg_shape)
            distribution = distribution(**dist_kwargs_b)
        
        super(TuneableScaledRateDistribution, self).__init__(distribution=distribution, bijector=bij)