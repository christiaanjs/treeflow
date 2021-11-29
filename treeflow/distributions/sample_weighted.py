import tensorflow as tf
from tensorflow_probability.python import distributions
from tensorflow_probability.python.distributions.distribution import Distribution
from tensorflow_probability.python.distributions.sample import Sample
from tensorflow_probability.python.internal import prefer_static as ps


class SampleWeighted(Sample):
    def __init__(
        self,
        distribution: Distribution,
        weights: tf.Tensor,
        sample_shape=(),
        validate_args=False,
        experimental_use_kahan_sum=False,
        name=None,
    ):
        """
        Parameters
        ----------
        weights: Tensor
            Tensor with shape `sample_shape`
        """
        self.weights = weights
        super().__init__(
            distribution=distribution,
            sample_shape=sample_shape,
            validate_args=validate_args,
            experimental_use_kahan_sum=experimental_use_kahan_sum,
            name=name,
        )

    def _finish_log_prob(self, lp, aux):
        (sample_ndims, extra_sample_ndims, batch_ndims) = aux

        # (1) Ensure lp is fully broadcast in the sample dims, i.e. ensure lp has
        #     full sample shape in the sample axes, before we reduce.
        bcast_lp_shape = ps.broadcast_shape(
            ps.shape(lp),
            ps.concat(
                [
                    ps.ones([sample_ndims], tf.int32),
                    ps.reshape(self.sample_shape, shape=[-1]),
                    ps.ones([batch_ndims], tf.int32),
                ],
                axis=0,
            ),
        )
        lp = tf.broadcast_to(lp, bcast_lp_shape)
        # (2) Make the final reduction.
        axis = ps.range(sample_ndims, sample_ndims + extra_sample_ndims)
        return self._sum_fn()(lp * self.weights, axis=axis)
