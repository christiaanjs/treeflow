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


class LeafCTMC(Distribution):
    def __init__(
        self,
        transition_probs_tree: TensorflowUnrootedTree,
        frequencies: tf.Tensor,
        validate_args=False,
        allow_nan_stats=True,
        name="LeafCTMC",
    ):
        parameters = dict(locals())
        super().__init__(
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            dtype=tf.int32,
            reparameterization_type=NOT_REPARAMETERIZED,
            parameters=parameters,
        )
        self.leaf_count = transition_probs_tree.taxon_count
        self.transition_probs_tree = transition_probs_tree
        self.frequencies = frequencies

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
            [self.leaf_count, tf.shape(self.transition_probs_tree.branch_lengths)[-1]]
        )

    def _event_shape_tensor(self) -> tf.Tensor:
        return tf.concat(
            [self.leaf_count, tf.shape(self.transition_probs_tree.branch_lengths)[-1]],
            axis=0,
        )

    def _sample_n(self, n, seed=None):
        raise NotImplemented("Sequence simulation not yet implemented")

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

    def _prob(self, x, seed=None):
        # TODO: Handle topology batch dims
        sample_and_batch_shape = ps.shape(x)[:-2]
        transition_probs = self._broadcast_transition_probs(sample_and_batch_shape)
        if self.transition_probs_tree.topology.has_batch_dimensions():
            raise NotImplementedError("Topology batching not yet supported")
        else:
            return phylogenetic_likelihood(
                x,
                transition_probs,
                self.frequencies,
                self.transition_probs_tree.topology.postorder_node_indices,
                self.transition_probs_tree.topology.node_child_indices,
                batch_shape=sample_and_batch_shape,
            )


__all__ = [LeafCTMC.__name__]
