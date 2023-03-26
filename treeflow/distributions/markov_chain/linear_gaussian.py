import tensorflow as tf
from functools import reduce
from treeflow.tree.topology.tensorflow_tree_topology import (
    TensorflowTreeTopology,
)
from treeflow.distributions.markov_chain.postorder import PostorderNodeMarkovChain
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.distributions import Normal, Independent


class LinearGaussianPostorderNodeMarkovChain(PostorderNodeMarkovChain):
    def __init__(
        self, topology: TensorflowTreeTopology, loc: tf.Tensor, scale: tf.Tensor
    ):
        """
        Parameters
        ----------
        loc
            Tensor with at least one dimension (node dimension on right)
        scale
            Tensor with at least one dimension (node dimension on right)
        """
        batch_and_event_shape = reduce(
            tf.broadcast_dynamic_shape,
            [
                tf.shape(loc),
                tf.shape(scale),
                tf.expand_dims(
                    tf.convert_to_tensor(topology.taxon_count, dtype=tf.int32) - 1, 0
                ),
            ],
        )
        self._batch_shape_tensor_value = batch_and_event_shape[:-1]
        batch_rank = tf.shape(batch_and_event_shape)[0] - 1
        self._loc_b = tf.broadcast_to(loc, batch_and_event_shape)
        loc_b_node_first = distribution_util.move_dimension(
            self._loc_b,
            batch_rank,
            0,
        )
        self._scale_b = tf.broadcast_to(scale, batch_and_event_shape)
        scale_b_node_first = distribution_util.move_dimension(
            self._scale_b,
            batch_rank,
            0,
        )

        super().__init__(
            topology,
            self._transition_fn,
            (loc_b_node_first, scale_b_node_first),
            childless_init=tf.zeros((0,), loc.dtype),
        )

    def _batch_shape_tensor(self):
        return self._batch_shape_tensor_value

    def _transition_fn(self, input, child_values):
        loc, scale = input
        return Normal(tf.reduce_sum(child_values, axis=0) + loc, scale)

    def _log_prob(self, x):
        """
        Vectorised log prob appears substantially faster than general TensorArray
        implementation in eager mode, similar in function mode.
        """
        batch_shape_tensor = self.batch_shape_tensor()
        batch_and_event_shape = tf.broadcast_dynamic_shape(
            tf.concat([batch_shape_tensor, self.event_shape_tensor()], axis=0),
            tf.shape(x),
        )

        x_b = tf.broadcast_to(x, batch_and_event_shape)
        dummy_leaf_values = tf.zeros(
            tf.concat(
                [batch_and_event_shape[:-1], [self._topology.taxon_count]], axis=0
            ),
            dtype=x.dtype,
        )
        with_leaf_values = tf.concat([dummy_leaf_values, x_b], axis=-1)
        child_values = tf.gather(
            with_leaf_values, self._topology.node_child_indices, axis=-1
        )
        child_sums = tf.reduce_sum(child_values, axis=-1)
        return Independent(
            Normal(self._loc_b + child_sums, self._scale_b), reinterpreted_batch_ndims=1
        ).log_prob(x_b)

    def _ragged_log_prob(self, x):
        """
        Experimental implementation using ragged tensors to do vectorised gathering of child values.
        Appears to be slower than both TensorArray and "dummy leaf value" vectorised implementations.
        """
        batch_shape_tensor = self.batch_shape_tensor()
        batch_and_event_shape = tf.broadcast_dynamic_shape(
            tf.concat([batch_shape_tensor, self.event_shape_tensor()], axis=0),
            tf.shape(x),
        )

        x_b = tf.broadcast_to(x, batch_and_event_shape)
        taxon_count = self._topology.taxon_count
        child_indices = (
            tf.ragged.boolean_mask(
                self._topology.node_child_indices,
                self._topology.node_child_indices > taxon_count,
            )
            - taxon_count
        )
        child_values = tf.gather(
            x_b,
            child_indices,
            axis=-1,
        )
        child_sums = tf.reduce_sum(child_values, axis=-1).to_tensor()
        return Independent(
            Normal(self._loc_b + child_sums, self._scale_b), reinterpreted_batch_ndims=1
        ).log_prob(x_b)
