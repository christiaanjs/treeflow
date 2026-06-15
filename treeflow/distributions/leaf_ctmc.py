import typing as tp
import tensorflow as tf
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.util import ParameterProperties
from treeflow.tree.unrooted.tensorflow_unrooted_tree import TensorflowUnrootedTree
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from tensorflow_probability.python.distributions import (
    NOT_REPARAMETERIZED,
    Distribution,
)
from treeflow.traversal.phylo_likelihood import phylogenetic_likelihood
from treeflow.traversal.sample_ctmc import sample_ctmc_preorder


def native_acceleration_available() -> bool:
    """Return True if the native phylogenetic-likelihood op can be loaded."""
    try:
        from treeflow.acceleration.native import is_available

        return is_available()
    except Exception:
        return False


class LeafCTMC(Distribution):
    def __init__(
        self,
        transition_probs_tree: TensorflowUnrootedTree,
        frequencies: tf.Tensor,
        validate_args=False,
        allow_nan_stats=True,
        use_native="auto",
        rescaling="adaptive",
        block_size=1,
        unroll="auto",
        name="LeafCTMC",
    ):
        parameters = dict(locals())
        super().__init__(
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            dtype=frequencies.dtype,
            reparameterization_type=NOT_REPARAMETERIZED,
            parameters=parameters,
        )
        self.leaf_count = transition_probs_tree.taxon_count
        self.transition_probs_tree = transition_probs_tree
        self.frequencies = frequencies
        # Keep the raw value (e.g. "auto") for serialization/parameters, and a
        # resolved boolean for dispatch. "auto" uses the native op when it is
        # available, otherwise falls back to the pure-TensorFlow implementation.
        self.use_native = use_native
        if use_native == "auto":
            self._use_native = native_acceleration_available()
        elif isinstance(use_native, bool):
            self._use_native = use_native
        else:
            raise ValueError(
                f"use_native must be True, False, or 'auto'; got {use_native!r}"
            )
        self.rescaling = rescaling
        # Site-blocking width for the native ops (SIMD); 1 = per-site (default).
        self.block_size = block_size
        # Unroll the (pure-TensorFlow) traversal for a static topology; see
        # treeflow.traversal.postorder.postorder_node_traversal.
        self.unroll = unroll

    @classmethod
    def _parameter_properties(
        ls, dtype, num_classes=None
    ) -> tp.Dict[str, ParameterProperties]:
        return dict(
            transition_probs_tree=ParameterProperties(
                event_ndims=TensorflowUnrootedTree(
                    branch_lengths=3, topology=TensorflowTreeTopology.get_event_ndims()
                )
            ),
            frequencies=ParameterProperties(event_ndims=1),
        )  # TODO: shape_fn

    def _event_shape(self) -> tf.TensorShape:
        return tf.TensorShape(
            [self.leaf_count, ps.shape(self.transition_probs_tree.branch_lengths)[-1]]
        )

    def _event_shape_tensor(self) -> tf.Tensor:
        return tf.concat(
            [self.leaf_count, tf.shape(self.transition_probs_tree.branch_lengths)[-1]],
            axis=0,
        )

    def _sample_n(self, n, seed=None):
        if self.transition_probs_tree.topology.has_batch_dimensions():
            raise NotImplementedError("Topology batching not yet supported")
        batch_shape = self.batch_shape_tensor()
        transition_probs = self._broadcast_transition_probs(
            tf.concat([[n], batch_shape], axis=0)
        )
        topology = self.transition_probs_tree.topology
        return sample_ctmc_preorder(
            transition_probs=transition_probs,
            frequencies=self.frequencies,
            preorder_indices=topology.preorder_indices,
            parent_indices=topology.parent_indices,
            taxon_count=topology.taxon_count,
            n=n,
            batch_shape=batch_shape,
            seed=seed,
        )

    def _broadcast_transition_probs(self, sample_and_batch_shape) -> tf.Tensor:
        transition_probs_shape = ps.shape(self.transition_probs_tree.branch_lengths)
        transition_probs_batch_shape = transition_probs_shape[:-3]
        additional_dims = (
            ps.shape(sample_and_batch_shape)[0]
            - ps.shape(transition_probs_batch_shape)[0]
        )
        new_shape = ps.concat(
            [ps.ones(additional_dims, dtype=tf.int32), transition_probs_shape], axis=0
        )
        return tf.reshape(self.transition_probs_tree.branch_lengths, new_shape)

    def _broadcast_for_likelihood(self, x):
        # TODO: Handle topology batch dims
        batch_shape = self.batch_shape_tensor()
        batch_and_event_shape = tf.concat(
            [batch_shape, self.event_shape_tensor()], axis=0
        )
        batch_and_event_rank = tf.shape(batch_and_event_shape)[0]
        sample_shape = tf.shape(x)[:-batch_and_event_rank]
        sample_and_batch_shape = tf.concat([sample_shape, batch_shape], axis=0)
        transition_probs = self._broadcast_transition_probs(sample_and_batch_shape)
        x_b = tf.broadcast_to(
            x, tf.concat([sample_shape, batch_and_event_shape], axis=0)
        )
        if self.transition_probs_tree.topology.has_batch_dimensions():
            raise NotImplementedError("Topology batching not yet supported")
        return x_b, transition_probs, sample_and_batch_shape

    def _prob(self, x, seed=None):
        x_b, transition_probs, sample_and_batch_shape = self._broadcast_for_likelihood(
            x
        )
        if self._use_native:
            from treeflow.acceleration.native import native_phylogenetic_likelihood

            return native_phylogenetic_likelihood(
                x_b,
                transition_probs,
                self.frequencies,
                self.transition_probs_tree.topology.postorder_node_indices,
                self.transition_probs_tree.topology.node_child_indices,
                batch_shape=sample_and_batch_shape,
                block_size=self.block_size,
            )
        return phylogenetic_likelihood(
            self.transition_probs_tree.topology,
            x_b,
            transition_probs,
            self.frequencies,
            batch_shape=sample_and_batch_shape,
            unroll=self.unroll,
        )

    def _log_prob(self, x, seed=None):
        if self.rescaling is False:
            # Preserve the default behaviour (log of the linear likelihood).
            return tf.math.log(self._prob(x))
        from treeflow.traversal.phylo_likelihood_dispatch import (
            phylogenetic_log_likelihood,
        )

        x_b, transition_probs, sample_and_batch_shape = self._broadcast_for_likelihood(
            x
        )
        return phylogenetic_log_likelihood(
            self.transition_probs_tree.topology,
            x_b,
            transition_probs,
            self.frequencies,
            batch_shape=sample_and_batch_shape,
            use_native=self._use_native,
            rescaling=self.rescaling,
            block_size=self.block_size,
            unroll=self.unroll,
        )


__all__ = ["LeafCTMC", "native_acceleration_available"]
