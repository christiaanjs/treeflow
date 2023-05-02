import numpy as np
import pytest
import attr
import tensorflow as tf
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from treeflow.model.approximation.cascading_flows import (
    DEFAULT_ACTIVATION_FUNCTIONS,
    get_trainable_highway_flow_parameters,
)
from treeflow.bijectors.highway_flow import (
    HighwayFlowParameters,
    HIGHWAY_FLOW_PARAMETER_EVENT_NDIMS,
)
from treeflow.bijectors.highway_flow_node_bijector import HighwayFlowNodeBijector


def test_highway_flow_node_bijector_shapes(hello_tensor_tree: TensorflowRootedTree):
    tree = hello_tensor_tree
    dtype = tree.node_heights.dtype
    activation_functions = DEFAULT_ACTIVATION_FUNCTIONS

    batch_shape = tf.stack([len(activation_functions), tree.taxon_count - 1])

    parameters = HighwayFlowParameters(
        **attr.asdict(
            get_trainable_highway_flow_parameters(
                2, batch_shape, dtype=dtype, defer=False
            )
        )
    )

    flow_bijector = HighwayFlowNodeBijector(
        tree.topology,
        parameters,
        (),
        flow_parameter_event_ndims=HIGHWAY_FLOW_PARAMETER_EVENT_NDIMS,
        activation_functions=activation_functions,
    )
    sample_shape = (4,)
    tree_shape = (tree.taxon_count,)
    event_shape = (1,)
    shape = sample_shape + tree_shape + event_shape
    base_value = tf.zeros(shape, dtype=dtype)
    forward = flow_bijector.forward(base_value)
    inverse = flow_bijector.inverse(forward)
    assert inverse.numpy().shape == shape
