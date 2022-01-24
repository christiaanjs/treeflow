from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.distributions.tree.base_tree_distribution import BaseTreeDistribution
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
import tensorflow as tf
from tensorflow_probability.python.internal import reparameterization


class RootedTreeDistribution(BaseTreeDistribution[TensorflowRootedTree]):
    def __init__(
        self,
        taxon_count,
        node_height_reparameterization_type: reparameterization.ReparameterizationType,
        sampling_time_reparameterization_type: reparameterization.ReparameterizationType,
        time_dtype=DEFAULT_FLOAT_DTYPE_TF,
        validate_args=False,
        allow_nan_stats=True,
        name="RootedTreeDistribution",
        parameters=None,
    ):
        super().__init__(
            taxon_count,
            reparameterization_type=TensorflowRootedTree(
                node_heights=node_height_reparameterization_type,
                sampling_times=sampling_time_reparameterization_type,
                topology=BaseTreeDistribution._topology_reparameterization_type,
            ),
            dtype=TensorflowRootedTree(
                node_heights=time_dtype,
                sampling_times=time_dtype,
                topology=BaseTreeDistribution._topology_dtype,
            ),
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            parameters=parameters,
        )

    def _event_shape(self) -> TensorflowRootedTree:
        taxon_count = self.taxon_count
        return TensorflowRootedTree(
            node_heights=tf.TensorShape([taxon_count - 1]),
            sampling_times=tf.TensorShape([taxon_count]),
            topology=self._topology_event_shape(),
        )

    def _event_shape_tensor(self) -> TensorflowRootedTree:
        taxon_count = tf.expand_dims(tf.convert_to_tensor(self.taxon_count), 0)
        return TensorflowRootedTree(
            node_heights=taxon_count - 1,
            sampling_times=taxon_count,
            topology=self._topology_event_shape_tensor(),
        )


__all__ = [RootedTreeDistribution.__name__]
