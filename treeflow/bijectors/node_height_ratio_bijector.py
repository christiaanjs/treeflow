import typing as tp
from numpy import block
import tensorflow as tf
from treeflow.traversal.ratio_transform import ratios_to_node_heights
from tensorflow_probability.python.bijectors import (
    Bijector,
    Chain,
    Blockwise,
    Sigmoid,
    Exp,
)
from tensorflow_probability.python.internal import tensor_util
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow import DEFAULT_FLOAT_DTYPE_TF


def native_ratio_transform_available() -> bool:
    """Return True if the native node-height ratio transform op can be loaded."""
    try:
        from treeflow.acceleration.native import ratio_transform_is_available

        return ratio_transform_is_available()
    except Exception:
        return False


class NodeHeightRatioBijector(Bijector):
    anchor_heights: tf.Tensor
    node_parent_indices: tf.Tensor

    def __init__(
        self,
        topology: TensorflowTreeTopology,
        anchor_heights: tp.Optional[tf.Tensor] = None,
        name="NodeHeightRatioBijector",
        validate_args=False,
        use_native="auto",
        unroll="auto",
    ):
        parameters = locals()
        self.topology = topology
        self.taxon_count = topology.taxon_count
        self.node_parent_indices = (
            self.topology.parent_indices[self.taxon_count :] - self.taxon_count
        )
        # Unroll the (pure-TensorFlow) ratio traversal for a static topology; see
        # treeflow.traversal.preorder.preorder_traversal.
        self.unroll = unroll

        # Resolve the forward-transform engine. The default ``"auto"`` uses the
        # native C++ op when it is built and falls back to the pure-TensorFlow
        # traversal otherwise; ``True``/``False`` force the choice. Only the
        # forward transform is accelerated; the inverse and log-det-Jacobian
        # remain pure TensorFlow. Note the native op registers only a first-order
        # gradient, so code that needs higher-order derivatives through the
        # forward transform (e.g. differentiating its log-det-Jacobian) should
        # pass ``use_native=False``.
        self.use_native = use_native
        if use_native == "auto":
            self._use_native = native_ratio_transform_available()
        elif isinstance(use_native, bool):
            self._use_native = use_native
        else:
            raise ValueError(
                f"use_native must be True, False, or 'auto'; got {use_native!r}"
            )

        if anchor_heights is None:
            self.anchor_heights = tf.zeros(
                topology.taxon_count - 1, dtype=DEFAULT_FLOAT_DTYPE_TF
            )
        else:
            self.anchor_heights = tensor_util.convert_nonref_to_tensor(anchor_heights)
        super().__init__(
            validate_args=validate_args,
            dtype=self.anchor_heights.dtype,
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
            name=name,
            parameters=parameters,
        )

    def _forward(self, ratios: tf.Tensor) -> tf.Tensor:
        if self.topology.has_batch_dimensions():
            raise NotImplementedError("Topology batching not yet supported")
        preorder_node_indices = (
            self.topology.preorder_node_indices - self.taxon_count
        )
        if self._use_native:
            from treeflow.acceleration.native import native_ratios_to_node_heights

            return native_ratios_to_node_heights(
                preorder_node_indices,
                self.node_parent_indices,
                ratios,
                self.anchor_heights,
            )
        return ratios_to_node_heights(
            self.topology, ratios, self.anchor_heights, unroll=self.unroll
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


class NodeHeightRatioChainBijector(Chain):
    def __init__(
        self,
        topology: TensorflowTreeTopology,
        anchor_heights: tp.Optional[tf.Tensor] = None,
        name="NodeHeightRatioChainBijector",
        validate_args=False,
        use_native="auto",
        unroll="auto",
    ):
        parameters = locals()
        height_bijector = NodeHeightRatioBijector(
            topology,
            anchor_heights,
            name=name,
            validate_args=validate_args,
            use_native=use_native,
            unroll=unroll,
        )
        blockwise_bijector = Blockwise(
            [Sigmoid(), Exp()], block_sizes=[topology.taxon_count - 2, 1]
        )
        super().__init__(
            [height_bijector, blockwise_bijector],
            parameters=parameters,
            name=name,
            validate_args=validate_args,
        )
