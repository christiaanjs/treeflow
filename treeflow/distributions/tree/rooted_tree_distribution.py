from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.distributions.tree.base_tree_distribution import BaseTreeDistribution
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
import tensorflow as tf
from tensorflow_probability.python.internal import reparameterization


class RootedTreeDistribution(BaseTreeDistribution[TensorflowRootedTree]):
    def __init__(
        self,
        taxon_count,
        height_reparameterization_type: reparameterization.ReparameterizationType,
        height_dtype=DEFAULT_FLOAT_DTYPE_TF,
        validate_args=False,
        allow_nan_stats=True,
        name="RootedTreeDistribution",
        parameters=None,
    ):
        super().__init__(
            taxon_count,
            reparameterization_type=TensorflowRootedTree(
                heights=height_reparameterization_type,
                topology=BaseTreeDistribution._topology_reparameterization_type,
            ),
            dtype=TensorflowRootedTree(
                heights=height_dtype, topology=BaseTreeDistribution._topology_dtype
            ),
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            parameters=parameters,
        )

    def _event_shape(self) -> TensorflowRootedTree:
        taxon_count = self.taxon_count
        return TensorflowRootedTree(
            heights=tf.TensorShape([2 * taxon_count - 1]),
            topology=self._topology_event_shape(),
        )

    def _event_shape_tensor(self) -> TensorflowRootedTree:
        taxon_count = tf.expand_dims(tf.convert_to_tensor(self.taxon_count), 0)
        return TensorflowRootedTree(
            heights=taxon_count * 2 - 1, topology=self._topology_event_shape_tensor()
        )


__all__ = [RootedTreeDistribution.__name__]
