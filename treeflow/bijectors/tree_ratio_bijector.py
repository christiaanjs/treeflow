import typing as tp
import tensorflow as tf
from tensorflow_probability.python.bijectors.bijector import Bijector
from tensorflow_probability.python.bijectors.identity import Identity, _NoOpCache
from tensorflow_probability.python.bijectors import JointMap
from treeflow.tree.topology.tensorflow_tree_topology import (
    TensorflowTreeTopology,
)
from treeflow.tree.rooted.tensorflow_rooted_tree import (
    TensorflowRootedTree,
    TensorflowRootedTreeAttrs,
)
from treeflow.bijectors.node_height_ratio_bijector import NodeHeightRatioBijector
from treeflow.distributions.tree.base_tree_distribution import BaseTreeDistribution


class TopologyIdentityBijector(Identity):
    def __init__(
        self,
        name="TopologyIdentityBijector",
        validate_args=False,
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

        # Override superclass private fields to eliminate caching, avoiding a memory
        # leak caused by the `y is x` characteristic of this bijector.
        self._from_x = self._from_y = _NoOpCache()


class RootedTreeBijector(JointMap):
    def __init__(
        self,
        height_bijector: Bijector,
        name="RootedTreeBijector",
        validate_args=False,
    ):
        super().__init__(
            TensorflowRootedTree(
                heights=height_bijector, topology=TopologyIdentityBijector()
            ),
            name=name,
            validate_args=validate_args,
        )
        # self._dtype = TensorflowRootedTree(
        #     heights=height_bijector.dtype, topology=BaseTreeDistribution._topology_dtype
        # )


class TreeRatioBijector(RootedTreeBijector):
    def __init__(
        self,
        topology: TensorflowTreeTopology,
        anchor_heights: tp.Optional[tf.Tensor] = None,
        name="TreeRatioBijector",
        validate_args=False,
    ):
        height_bijector = NodeHeightRatioBijector(topology, anchor_heights)
        super().__init__(
            height_bijector=height_bijector,
            name=name,
            validate_args=validate_args,
        )
