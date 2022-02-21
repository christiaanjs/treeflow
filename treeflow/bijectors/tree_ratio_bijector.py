import typing as tp
import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from tensorflow_probability.python.bijectors.bijector import Bijector
from tensorflow_probability.python.bijectors.identity import Identity, _NoOpCache
from tensorflow_probability.python.bijectors import JointMap, Chain
from treeflow.tree.topology.tensorflow_tree_topology import (
    TensorflowTreeTopology,
)
from treeflow.tree.rooted.tensorflow_rooted_tree import (
    TensorflowRootedTree,
    TensorflowRootedTreeAttrs,
)
from treeflow.bijectors.node_height_ratio_bijector import NodeHeightRatioChainBijector
from treeflow.distributions.tree.base_tree_distribution import BaseTreeDistribution
from tensorflow_probability.python.bijectors import softplus as softplus_bijector


class TopologyIdentityBijector(Identity):
    def __init__(
        self,
        name="TopologyIdentityBijector",
        validate_args=False,
        prob_dtype=DEFAULT_FLOAT_DTYPE_TF,
    ):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super(Identity, self).__init__(
                forward_min_event_ndims=TensorflowTreeTopology.get_event_ndims(),
                dtype=tf.int32,
                is_constant_jacobian=True,
                validate_args=validate_args,
                parameters=parameters,
                name=name,
            )
        self.prob_dtype = prob_dtype

        # Override superclass private fields to eliminate caching, avoiding a memory
        # leak caused by the `y is x` characteristic of this bijector.
        self._from_x = self._from_y = _NoOpCache()

    def _inverse_log_det_jacobian(self, y):
        return tf.constant(0, dtype=self.prob_dtype)


class RootedTreeBijector(JointMap):
    def __init__(
        self,
        node_height_bijector: Bijector,
        sampling_time_bijector: Bijector,
        name="RootedTreeBijector",
        validate_args=False,
    ):
        super().__init__(
            TensorflowRootedTree(
                node_heights=node_height_bijector,
                sampling_times=sampling_time_bijector,
                topology=TopologyIdentityBijector(),
            ),
            name=name,
            validate_args=validate_args,
        )


class TreeRatioBijector(RootedTreeBijector):
    def __init__(
        self,
        topology: TensorflowTreeTopology,
        anchor_heights: tp.Optional[tf.Tensor] = None,
        fixed_sampling_times: bool = True,
        name="TreeRatioBijector",
        validate_args=False,
    ):
        height_bijector = NodeHeightRatioChainBijector(topology, anchor_heights)
        sampling_time_bijector = (
            Identity()
            if fixed_sampling_times
            else softplus_bijector.Softplus(validate_args=validate_args)
        )
        super().__init__(
            node_height_bijector=height_bijector,
            sampling_time_bijector=sampling_time_bijector,
            name=name,
            validate_args=validate_args,
        )


__all__ = [RootedTreeBijector.__name__, TreeRatioBijector.__name__]
