import typing as tp
from functools import reduce
import tensorflow as tf
from tensorflow_probability.python.distributions import Distribution
from tensorflow_probability.python.internal import samplers
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.traversal.postorder import postorder_node_traversal, PostorderTopologyData
from tensorflow_probability.python.internal import distribution_util
import tensorflow_probability.python.internal.prefer_static as ps


class PostorderNodeMarkovChain(Distribution):
    def __init__(
        self,
        topology: TensorflowTreeTopology,
        transition_fn: tp.Callable[[object, object], Distribution],
        dist_input: object,
        childless_init: tp.Optional[object] = None,
        validate_args=False,
        allow_nan_stats=True,
        name="PostorderNodeMarkovChain",
    ):
        """
        Parameters
        ----------
        topology

        transition_fn
            Callable with arguments (input, child values)
            Must get batch shape from child values

        dist_input
            Structure of Tensors that are parameters of `transition_fn`.
            Nodes must be on the first axis.

        childless_init
            Structure of Tensors that represent the sample input to a node
            with only leaf children.
        """
        self._transition_fn = transition_fn
        self._topology = topology
        self._leaf_parent_mask = topology.child_indices
        self._dist_input = dist_input

        concrete_dist = self._transition_fn(
            tf.nest.map_structure(lambda x: tf.gather(x, 0), self._dist_input),
            [] if childless_init is None else childless_init,
        )
        self._dist_event_shape_tensor = concrete_dist.event_shape_tensor
        self._dist_event_shape = concrete_dist.event_shape
        self._dist_event_rank = tf.nest.map_structure(
            lambda event_shape: ps.rank_from_shape(event_shape),
            self._dist_event_shape,
        )
        self._dist_dtype = concrete_dist.dtype
        self._dist_batch_shape = concrete_dist.batch_shape
        self._dist_log_prob_dtype = concrete_dist.log_prob(concrete_dist.sample()).dtype
        super().__init__(
            self._dist_dtype,
            concrete_dist.reparameterization_type,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
        )

    def _event_shape_tensor(self):
        taxon_count = self._topology.taxon_count
        return tf.nest.map_structure(
            lambda shape: tf.concat([[taxon_count - 1], shape], axis=0),
            self._dist_event_shape_tensor,
        )

    def _event_shape(self):
        taxon_count = self._topology.taxon_count
        return tf.nest.map_structure(
            lambda shape: tf.TensorShape([taxon_count - 1]).concatenate(shape),
            self._dist_event_shape,
        )

    def _sample_n(self, n, seed=None):
        batch_shape = self.batch_shape_tensor()
        taxon_count = self._topology.taxon_count
        traversal_seeds = samplers.split_seed(
            seed=seed,
            n=tf.convert_to_tensor(self._topology.taxon_count, dtype=tf.int32) - 1,
        )

        def sample_mapping(child_samples, input, topology_data: PostorderTopologyData):
            node_seed, dist_input = input
            child_node_mask = topology_data.child_indices >= taxon_count
            node_child_samples = tf.nest.map_structure(
                lambda x: tf.ragged.boolean_mask(x, child_node_mask), child_samples
            )
            node_dist = self._transition_fn(dist_input, node_child_samples)
            return node_dist.sample(
                seed=node_seed
            )  # Sample shape passed in as batch shape

        dummy_leaf_init = tf.nest.map_structure(
            lambda shape, dtype: tf.zeros(
                tf.concat([[taxon_count], [n], batch_shape, shape], 0), dtype
            ),
            self._dist_event_shape,
            self._dist_dtype,
        )

        traversal_result = postorder_node_traversal(
            self._topology,
            sample_mapping,
            (traversal_seeds, self._dist_input),
            dummy_leaf_init,
        )
        node_only_result = tf.nest.map_structure(
            lambda x: x[taxon_count:], traversal_result
        )
        transposed = tf.nest.map_structure(
            lambda x, event_rank: distribution_util.move_dimension(
                x, 0, ps.rank(x) - event_rank - 1
            ),
            node_only_result,
            self._dist_event_rank,
        )
        return transposed

    def _batch_shape(self):
        return self._dist_batch_shape

    def _log_prob(self, x):
        taxon_count = self._topology.taxon_count
        dist_event_rank = self._dist_event_rank
        x_batch_shape = tf.nest.map_structure(
            lambda element_event_rank, element: tf.shape(element)[
                : -(element_event_rank + 1)
            ],
            dist_event_rank,
            x,
        )
        x_batch_shape_flat = tf.nest.flatten(x_batch_shape)
        x_batch_shape_reduced = (
            x_batch_shape_flat[0]
            if len(x_batch_shape_flat) == 1
            else reduce(tf.broadcast_dynamic_shape, x_batch_shape_flat)
        )
        batch_shape = tf.broadcast_dynamic_shape(
            x_batch_shape_reduced, self.batch_shape
        )
        batch_rank = tf.shape(batch_shape)[0]

        x_b = tf.nest.map_structure(
            lambda element, event_shape_element: tf.broadcast_to(
                element, tf.concat([batch_shape, event_shape_element], 0)
            ),
            x,
            self.event_shape,
        )
        x_node_first = tf.nest.map_structure(
            lambda element: distribution_util.move_dimension(element, batch_rank, 0),
            x_b,
        )
        # all_node_values = tf.concat([dummy_leaf_init, x_b], axis=batch_rank)

        dummy_leaf_log_probs = tf.zeros(
            tf.concat([[taxon_count], batch_shape], 0), self._dist_log_prob_dtype
        )

        def log_prob_mapping(
            _, input, topology_data: PostorderTopologyData
        ) -> tf.Tensor:
            dist_input, node_value = input
            child_node_mask = topology_data.child_indices >= taxon_count
            node_child_indices = (
                tf.ragged.boolean_mask(topology_data.child_indices, child_node_mask)
                - taxon_count
            )
            child_values = tf.nest.map_structure(
                lambda element: tf.gather(element, node_child_indices, axis=0),
                x_node_first,
            )
            dist = self._transition_fn(dist_input, child_values)
            return dist.log_prob(node_value)

        node_log_probs = postorder_node_traversal(
            self._topology,
            log_prob_mapping,
            (self._dist_input, x_node_first),
            dummy_leaf_log_probs,
        )
        return tf.reduce_sum(node_log_probs[taxon_count:], 0)
