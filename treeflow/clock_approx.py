
import tensorflow as tf
import tensorflow_probability as tfp
import treeflow.sequences

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