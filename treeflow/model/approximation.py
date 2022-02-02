import tensorflow as tf
import tensorflow_probability.python.distributions as tfd


def get_base_distribution(flat_event_size):
    base_standard_dist = tfd.JointDistributionSequential(
        [tfd.Sample(tfd.Normal(loc=0.0, scale=1.0), s) for s in flat_event_size]
    )
    return base_standard_dist


def get_mean_field_operators(flat_event_size):
    pass


def get_trainable_shift_bijector(flat_event_size):
    pass


def get_mean_field_approximation():

    flat_event_size = tf.nest.map_structure(
        tf.reduce_prod, tf.nest.flatten(event_shape_tensor)
    )
