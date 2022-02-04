import typing as tp
import tensorflow as tf
from tensorflow_probability.python.bijectors.bijector import Bijector
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology


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
        super().__init__(name=name, validate_args=validate_args)

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
