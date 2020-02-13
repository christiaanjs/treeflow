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

    def _call_sample_n(self, sample_shape, seed, name, **kwargs):
        with self._name_and_control_scope(name):
            sample_shape = tf.cast(sample_shape, tf.int32, name='sample_shape')
            sample_shape, n = self._expand_sample_shape_to_vector(sample_shape, 'sample_shape')
            tree_samples = self._sample_n(n, seed)

            def reshape_samples(samples):
                batch_event_shape = tf.shape(samples)[1:]
                final_shape = tf.concat([sample_shape, batch_event_shape], 0)
                samples = tf.reshape(samples, final_shape)
                samples = self._set_sample_static_shape(samples, sample_shape)
                return samples

            return {
                'heights': reshape_samples(tree_samples['heights']),
                'topology': {
                    'parent_indices': reshape_samples(tree_samples['topology']['parent_indices'])
                }
            }