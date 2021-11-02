import typing as tp

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.util import ParameterProperties
from treeflow.tree.unrooted.tensorflow_unrooted_tree import TensorflowUnrootedTree
from tensorflow_probability.python.distributions import (
    NOT_REPARAMETERIZED,
    Distribution,
)


class LeafCTMC(Distribution):
    pass
    # TODO: Treat sites as batch dim

    def __init__(
        self,
        transition_probs_tree: TensorflowUnrootedTree,
        validate_args=False,
        allow_nan_stats=True,
        name="LeafCTMC",
    ):
        super().__init__(
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            dtype=tf.int32,
            reparameterization_type=NOT_REPARAMETERIZED,
        )
        self.leaf_count = transition_probs_tree.taxon_count
        self.transition_probs_tree = transition_probs_tree

    @classmethod
    def _parameter_properties(
        ls, dtype, num_classes=None
    ) -> tp.Dict[str, ParameterProperties]:
        return dict(
            transition_probs_tree=ParameterProperties(event_ndims=3)
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

    def _log_prob(self, n, seed=None):
        raise NotImplemented("Phylogenetic likelihood not yet implemented")


__all__ = [LeafCTMC.__name__]
