import tensorflow as tf
import tensorflow_probability as tfp

class TreeDistribution(tfp.distributions.Distribution):
    def __init__(self, **kwargs):
        super(TreeDistribution, self).__init__(
            dtype={
                'heights': tf.float32,
                'topology': {
                    'parent_indices': tf.int32
                }
            },
            reparameterization_type=tfp.distributions.NOT_REPARAMETERIZED,
            **kwargs
        )

    # Borrwoed from JointDistribution
    # We need to bypass base Distribution reshaping/validation logic so we
    # tactically implement a few of the `_call_*` redirectors. 
    def _call_log_prob(self, value, name):
        with self._name_and_control_scope(name):
            return self._log_prob(value)

    def _call_sample_n(self, sample_shape, seed, name):
        with self._name_and_control_scope(name):
            return self._sample_n(sample_shape, seed)