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
from treeflow.traversal.phylo_likelihood import phylogenetic_log_likelihood_rescaled


class RescaledLeafCTMC(Distribution):
    """
    Phylogenetic leaf distribution using rescaled partial likelihoods.

    Identical interface to :class:`~treeflow.distributions.leaf_ctmc.LeafCTMC`
    but implements ``_log_prob`` directly via
    :func:`~treeflow.traversal.phylo_likelihood.phylogenetic_log_likelihood_rescaled`,
    which rescales partial likelihoods at every internal node to prevent
    numerical underflow on large trees or with very small branch lengths.
    """

    def __init__(
        self,
        transition_probs_tree: TensorflowUnrootedTree,
        frequencies: tf.Tensor,
        validate_args=False,
        allow_nan_stats=True,
        name="RescaledLeafCTMC",
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

    @classmethod
    def _parameter_properties(
        cls, dtype, num_classes=None
    ) -> tp.Dict[str, ParameterProperties]:
        return dict(
            transition_probs_tree=ParameterProperties(
                event_ndims=TensorflowUnrootedTree(
                    branch_lengths=3, topology=TensorflowTreeTopology.get_event_ndims()
                )
            ),
            frequencies=ParameterProperties(event_ndims=1),
        )

    def _event_shape(self) -> tf.TensorShape:
        return tf.TensorShape(
            [self.leaf_count, ps.shape(self.transition_probs_tree.branch_lengths)[-1]]
        )

    def _event_shape_tensor(self) -> tf.Tensor:
        return tf.concat(
            [self.leaf_count, tf.shape(self.transition_probs_tree.branch_lengths)[-1]],
            axis=0,
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

    def _log_prob(self, x, seed=None):
        if self.transition_probs_tree.topology.has_batch_dimensions():
            raise NotImplementedError("Topology batching not yet supported")
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
        return phylogenetic_log_likelihood_rescaled(
            x_b,
            transition_probs,
            self.frequencies,
            self.transition_probs_tree.topology.postorder_node_indices,
            self.transition_probs_tree.topology.node_child_indices,
            batch_shape=sample_and_batch_shape,
        )


__all__ = ["RescaledLeafCTMC"]
