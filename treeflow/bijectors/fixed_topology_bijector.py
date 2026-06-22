import typing as tp
import tensorflow as tf
from tensorflow_probability.python.bijectors.bijector import Bijector
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.distributions.tree.base_tree_distribution import BaseTreeDistribution


class FixedTopologyRootedTreeBijector(Bijector):
    def __init__(
        self,
        topology,
        height_bijector: Bijector,
        sampling_times: tp.Optional[tf.Tensor] = None,
        name="FixedTopologyRootedTreeBijector",
        validate_args=False,
    ):
        # ``topology`` may be a TensorflowTreeTopology (tensor arrays) or a static
        # NumPy topology (e.g. StaticNumpyTreeTopology). The latter is kept as-is for
        # the height bijector (so its traversal can fold/unroll the static indices),
        # and rebuilt as an in-graph-constant TensorflowTreeTopology for the tree
        # value emitted by ``_forward`` (see ``_tensor_topology``).
        self.topology = topology
        self.height_bijector = height_bijector
        if sampling_times is None:
            self.sampling_times = tf.zeros(
                self.topology.taxon_count, dtype=height_bijector.forward_dtype()
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

    def _tensor_topology(self) -> TensorflowTreeTopology:
        """The topology as a TensorflowTreeTopology for the emitted tree value.

        A TensorflowTreeTopology pin is returned unchanged. A static NumPy topology is
        rebuilt with its index arrays as ``tf.constant``s via
        ``to_constant_tensor_topology``; since ``_forward`` runs inside the (VI) trace
        these are in-graph ``Const`` ops, so the JointDistribution sees a normal
        tensor topology *and* the downstream likelihood traversal can still fold them
        (``tf.get_static_value``) and unroll at any tree size.
        """
        topology = self.topology
        if isinstance(topology, TensorflowTreeTopology):
            return topology
        return topology.to_constant_tensor_topology()

    def _forward(self, x):
        node_heights = self.height_bijector.forward(x)
        return TensorflowRootedTree(
            topology=self._tensor_topology(),
            sampling_times=self.sampling_times,
            node_heights=node_heights,
        )

    def _inverse(self, tree):
        return self.height_bijector.inverse(tree.node_heights)

    def _log_det_jacobian(self, x):
        return self.height_bijector.log_det_jacobian(x)

    # def _inverse_log_det_jacobian(self, tree):
    #     return self.height_bijector.inverse_log_det_jacobian(tree.node_heights)

    def _call_forward_log_det_jacobian(self, x, event_ndims, name, **kwargs):
        return self.height_bijector.forward_log_det_jacobian(
            x, event_ndims=event_ndims, name=name, **kwargs
        )

    def _call_inverse_log_det_jacobian(self, y, event_ndims, name, **kwargs):
        return self.height_bijector.inverse_log_det_jacobian(
            y.node_heights, event_ndims=event_ndims.node_heights, name=name, **kwargs
        )

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

    def inverse_event_ndims(self, event_ndims, **kwargs):
        return self.height_bijector.inverse_event_ndims(event_ndims.node_heights)

    def forward_event_ndims(self, event_ndims, **kwargs):
        min_event_dims = tp.cast(TensorflowRootedTree, self.inverse_min_event_ndims)
        return min_event_dims.with_node_heights(
            self.height_bijector.forward_event_ndims(event_ndims)
        )
