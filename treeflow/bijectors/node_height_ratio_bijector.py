import typing as tp
import tensorflow as tf
from treeflow.traversal.ratio_transform import ratios_to_node_heights
from tensorflow_probability.python.bijectors.bijector import Bijector
from treeflow.tree import topology
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow import DEFAULT_FLOAT_DTYPE_TF


class NodeHeightRatioBijector(Bijector):
    anchor_heights: tf.Tensor
    node_parent_indices: tf.Tensor

    def __init__(
        self,
        topology: TensorflowTreeTopology,
        anchor_heights: tp.Optional[tf.Tensor] = None,
        name="NodeHeightRatioBijector",
        validate_args=False,
    ):
        self.topology = topology
        self.taxon_count = topology.taxon_count
        self.node_parent_indices = (
            self.topology.parent_indices[self.taxon_count :] - self.taxon_count
        )

        if anchor_heights is None:
            self.anchor_heights = tf.zeros(
                topology.taxon_count - 1, dtype=DEFAULT_FLOAT_DTYPE_TF
            )
        else:
            self.anchor_heights = anchor_heights
        super().__init__(
            validate_args=validate_args,
            dtype=self.anchor_heights.dtype,
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
            name=name,
        )

    def _forward(self, ratios: tf.Tensor) -> tf.Tensor:
        if self.topology.has_batch_dimensions():
            raise NotImplementedError("Topology batching not yet supported")
        else:
            return ratios_to_node_heights(
                self.topology.preorder_node_indices - self.taxon_count,
                self.node_parent_indices,
                ratios,
                self.anchor_heights,
            )

    def _inverse(self, heights: tf.Tensor) -> tf.Tensor:
        if self.topology.has_batch_dimensions():
            raise NotImplementedError("Topology batching not yet supported")
        else:
            minus_anchors = heights - self.anchor_heights
            ratios = minus_anchors[..., :-1] / (
                tf.gather(heights, self.node_parent_indices, axis=-1)
                - self.anchor_heights[..., :-1]
            )
            root_value = minus_anchors[..., -1:]
            return tf.concat([ratios, root_value], axis=-1)

    def _inverse_log_det_jacobian(self, heights: tf.Tensor) -> tf.Tensor:
        if self.topology.has_batch_dimensions():
            raise NotImplementedError("Topology batching not yet supported")
        else:
            return -tf.reduce_sum(
                tf.math.log(
                    tf.gather(heights, self.node_parent_indices, axis=-1)
                    - self.anchor_heights[..., :-1]
                ),
                axis=-1,
            )
