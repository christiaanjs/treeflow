import typing as tp
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
        """
        self._transition_fn = transition_fn
        self._topology = topology
        self._leaf_parent_mask = topology.child_indices
        self._dist_input = dist_input

        concrete_dist = self._transition_fn(
            tf.nest.map_structure(lambda x: tf.gather(x, 0), self._dist_input),
            [] if childless_init is None else childless_init,
        )
        self._dist_event_shape = concrete_dist.event_shape
        self._dist_event_rank = tf.nest.map_structure(
            lambda event_shape: ps.rank_from_shape(event_shape),
            self._dist_event_shape,
        )
        self._dist_dtype = concrete_dist.dtype
        super().__init__(
            self._dist_dtype,
            concrete_dist.reparameterization_type,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
        )

    def _sample_n(self, n, seed=None):
        # track seed

        taxon_count = self._topology.taxon_count
        traversal_seeds = samplers.split_seed(seed=seed, n=self._topology.taxon_count)

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
                tf.concat([[taxon_count], [n], shape], 0), dtype
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

    def _log_prob(self, x):
        pass
