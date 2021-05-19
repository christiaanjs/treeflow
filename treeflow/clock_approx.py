import tensorflow as tf
import tensorflow_probability as tfp
import treeflow.sequences
import treeflow


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
        self, distribution, tree, clock_rate, **dist_kwargs
    ):  # TODO: dist_kwargs_event_ndims for different kwrags
        blens = treeflow.sequences.get_branch_lengths(tree)
        batch_shape = tf.shape(clock_rate)
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


class TuneableScaledRateDistribution(tfp.distributions.TransformedDistribution):
    def __init__(
        self, distribution, scale_power, tree, clock_rate, **dist_kwargs
    ):  # TODO: dist_kwargs_event_ndims for different kwrags
        blens = treeflow.sequences.get_branch_lengths(tree)
        batch_shape = tf.shape(clock_rate)
        bij = tfp.bijectors.Scale(
            1.0 / tf.math.pow(blens * tf.expand_dims(clock_rate, -1), scale_power)
        )

        if callable(distribution):
            dist_kwargs_b = {}
            for key, val in dist_kwargs.items():
                arg_shape = tf.concat([batch_shape, tf.shape(val)], 0)
                dist_kwargs_b[key] = tf.broadcast_to(val, arg_shape)
            distribution = distribution(**dist_kwargs_b)

        super(TuneableScaledRateDistribution, self).__init__(
            distribution=distribution, bijector=bij
        )


def get_lognormal_loc_conjugate_posterior(loc_prior):
    assert isinstance(loc_prior, tfp.distributions.Normal)

    def conjugate_posterior(rates):
        n = tf.cast(tf.shape(rates)[-1], treeflow.DEFAULT_FLOAT_DTYPE_TF)
        log_rates_mean = tf.reduce_mean(tf.math.log(tf.rates), axis=-1)
        loc_posterior_loc = (loc_prior.scale * loc_prior.loc + n * log_rates_mean) / (
            loc_prior.scale + n
        )
        loc_posterior_scale = loc_prior.scale + n
        return tfp.distributions.Normal(
            loc=loc_posterior_loc, scale=loc_posterior_scale
        )

    return conjugate_posterior


def get_lognormal_precision_conjugate_posterior(loc_prior, precision_prior):
    assert isinstance(precision_prior, tfp.distributions.Gamma)

    def conjugate_posterior(rates):
        n = tf.cast(tf.shape(rates)[-1], treeflow.DEFAULT_FLOAT_DTYPE_TF)
        log_rates = tf.math.log(tf.rates)
        log_rates_mean = tf.reduce_mean(log_rates, axis=-1)
        log_rates_sum_of_squares = tf.reduce_sum(
            tf.square(log_rates - tf.expand_dims(log_rates_mean, -1)), axis=-1
        )
        precision_posterior_concentration = (
            precision_prior.concentration + n / 2.0
        )  # alpha
        precision_posterior_rate = (
            precision_prior.rate
            + 0.5 * log_rates_sum_of_squares
            + (n * loc_prior.scale)
            / (loc_prior.scale + n)
            * tf.square(log_rates_mean - loc_prior.loc)
            / 2.0
        )  # beta
        return tfp.distributions.Gamma(
            concentration=precision_posterior_concentration,
            rate=precision_posterior_rate,
        )

    return conjugate_posterior
