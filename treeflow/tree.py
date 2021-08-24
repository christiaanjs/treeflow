import tensorflow as tf
import tensorflow_probability as tfp
from treeflow import DEFAULT_FLOAT_DTYPE_TF, DEFAULT_FLOAT_DTYPE_NP


class TreeDistribution(tfp.distributions.Distribution):
    def __init__(self, taxon_count, **kwargs):
        super(TreeDistribution, self).__init__(
            dtype={
                "heights": DEFAULT_FLOAT_DTYPE_TF,
                "topology": {"parent_indices": tf.int32},
            },
            reparameterization_type=tfp.distributions.NOT_REPARAMETERIZED,
            **kwargs,
        )
        self._taxon_count = taxon_count

    def _event_shape(self):
        return {
            "heights": tf.TensorShape([2 * self._taxon_count - 1]),
            "topology": {"parent_indices": tf.TensorShape([2 * self._taxon_count - 2])},
        }

    def _event_shape_tensor(self):
        taxon_count_tensor = tf.convert_to_tensor(self._taxon_count)
        return {
            "heights": tf.TensorShape([2 * taxon_count_tensor - 1]),
            "topology": {
                "parent_indices": tf.TensorShape([2 * taxon_count_tensor - 2])
            },
        }

    # Borrwoed from JointDistribution
    # We need to bypass base Distribution reshaping/validation logic so we
    # tactically implement a few of the `_call_*` redirectors.
    def _call_log_prob(self, value, name):
        with self._name_and_control_scope(name):
            return self._log_prob(value)

    def _call_sample_n(self, sample_shape, seed, name, **kwargs):

        with self._name_and_control_scope(name):
            sample_shape = tf.cast(sample_shape, tf.int32, name="sample_shape")
            sample_shape, n = self._expand_sample_shape_to_vector(
                sample_shape, "sample_shape"
            )
            tree_samples = self._sample_n(
                n,
                seed=seed() if callable(seed) else seed,
                **kwargs,
            )

            def reshape_samples(samples):
                batch_event_shape = tf.shape(samples)[1:]
                final_shape = tf.concat([sample_shape, batch_event_shape], 0)
                samples = tf.reshape(samples, final_shape)
                # samples = self._set_sample_static_shape(samples, sample_shape)
                return samples

            return {
                "heights": reshape_samples(tree_samples["heights"]),
                "topology": {
                    "parent_indices": reshape_samples(
                        tree_samples["topology"]["parent_indices"]
                    )
                },
            }
