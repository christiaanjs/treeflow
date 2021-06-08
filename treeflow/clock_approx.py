import tensorflow as tf
import tensorflow_probability as tfp
import treeflow.sequences
import treeflow
from treeflow.priors import precision_to_scale


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

        super(ScaledDistribution, self).__init__(
            distribution=distribution, bijector=bij
        )


class ScaledRateDistribution(tfp.distributions.TransformedDistribution):
    def __init__(
        self, distribution, tree, clock_rate=None, **dist_kwargs
    ):  # TODO: dist_kwargs_event_ndims for different kwrags
        blens = treeflow.sequences.get_branch_lengths(tree)
        batch_shape = tf.shape(blens)[:-1]
        if clock_rate is None:
            bij = tfp.bijectors.Scale(1.0 / blens)
        else:
            bij = tfp.bijectors.Scale(1.0 / (blens * tf.expand_dims(clock_rate, -1)))

        if callable(distribution):
            dist_kwargs_b = {}
            for key, val in dist_kwargs.items():
                arg_shape = tf.concat([batch_shape, tf.shape(val)], 0)
                dist_kwargs_b[key] = tf.broadcast_to(val, arg_shape)
            distribution = distribution(**dist_kwargs_b)

        super(ScaledRateDistribution, self).__init__(
            distribution=distribution, bijector=bij
        )


def get_normal_conjugate_posterior_dict(concentration, rate, loc, precision_scale):
    def precision_posterior(x):
        n = tf.cast(tf.shape(x)[-1], treeflow.DEFAULT_FLOAT_DTYPE_TF)
        posterior_concentration = concentration + n / 2.0

        sample_mean = tf.reduce_mean(x, axis=-1)
        sample_variance = tf.square(tf.math.reduce_std(x, axis=-1))
        posterior_rate = (
            rate
            + (
                n * sample_variance
                + (precision_scale * n * tf.square(sample_mean - loc))
                / (precision_scale + n)
            )
            / 2.0
        )
        return tfp.distributions.Gamma(
            concentration=posterior_concentration, rate=posterior_rate
        )

    def loc_posterior(x, precision):
        n = tf.cast(tf.shape(x)[-1], treeflow.DEFAULT_FLOAT_DTYPE_TF)
        sample_mean = tf.reduce_mean(x, axis=-1)
        posterior_loc = (precision_scale * loc + n * sample_mean) / (
            precision_scale + n
        )
        posterior_precision_scale = precision_scale + n
        return tfp.distributions.Normal(
            loc=posterior_loc,
            scale=precision_to_scale(posterior_precision_scale * precision),
        )

    return dict(precision=precision_posterior, loc=loc_posterior)
