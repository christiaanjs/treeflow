import typing as tp
from xml.dom import NOT_FOUND_ERR
import tensorflow as tf
from tensorflow_probability.python.bijectors.bijector import Bijector
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.distributions.tree.base_tree_distribution import BaseTreeDistribution


class FixedTopologyRootedTreeBijector(Bijector):
    def __init__(
        self,
        topology: TensorflowTreeTopology,
        height_bijector: Bijector,
        sampling_times: tp.Optional[tf.Tensor] = None,
        name="FixedTopologyRootedTreeBijector",
        validate_args=False,
    ):
        self.topology = topology
        self.height_bijector = height_bijector
        if sampling_times is None:
            self.sampling_times = tf.zeros(
                self.topology.taxon_count, dtype=height_bijector.dtype
            )
        else:
            self.sampling_times = sampling_times
        super().__init__(
            name=name,
            validate_args=validate_args,
            forward_min_event_ndims=1,
            inverse_min_event_ndims=TensorflowRootedTree(
                node_heights=1,
                sampling_times=1,
                topology=TensorflowTreeTopology.get_event_ndims(),
            ),
            dtype=self.height_bijector.dtype,
        )

    def _forward(self, x):
        node_heights = self.height_bijector.forward(x)
        return TensorflowRootedTree(
            topology=self.topology,
            sampling_times=self.sampling_times,
            node_heights=node_heights,
        )

    def _inverse(self, tree):
        return self.height_bijector.inverse(tree.node_heights)

    def _log_det_jacobian(self, x):
        return self.height_bijector.log_det_jacobian(x)

    def _inverse_log_det_jacobian(self, tree):
        return self.height_bijector.inverse_log_det_jacobian(tree.node_heights)

    def _forward_dtype(self, input_dtype, **kwargs):
        return TensorflowRootedTree(
            node_heights=input_dtype,
            sampling_times=self.sampling_times.dtype,
            topology=BaseTreeDistribution._topology_dtype,
        )

    def _forward_event_shape(self, input_shape):  # TODO: Get this logic from JointMap?
        return TensorflowRootedTree(
            node_heights=self.height_bijector.forward_event_shape(input_shape),
            sampling_times=self.sampling_times.shape,
            topology=BaseTreeDistribution._static_topology_event_shape(
                self.topology.taxon_count
            ),
        )

    def _forward_event_shape_tensor(self, input_shape):
        return TensorflowRootedTree(
            node_heights=self.height_bijector.forward_event_shape_tensor(input_shape),
            sampling_times=tf.shape(self.sampling_times),
            topology=BaseTreeDistribution._static_topology_event_shape_tensor(
                self.topology.taxon_count
            ),
        )

    def _inverse_dtype(self, output_dtype: TensorflowRootedTree, **kwargs):
        return output_dtype.node_heights

    def _inverse_event_shape(self, output_shape):
        return self.height_bijector.inverse_event_shape(output_shape.node_heights)

    def _inverse_event_shape_tensor(self, output_shape):
        return self.height_bijector.inverse_event_shape_tensor(
            output_shape.node_heights
        )
