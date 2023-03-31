import typing as tp
from functools import partial
import tensorflow as tf
from tensorflow_probability.python.bijectors import Bijector, Shift, Scale, Chain
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.bijectors.highway_flow import HighwayFlow, HighwayFlowParameters
from treeflow.bijectors.preorder_node_bijector import PreorderNodeBijector


def build_shift_and_scale_bijector(input: tf.Tensor):
    return Chain([Shift(input[..., 1]), Scale(input[..., 0])])


def highway_flow_bijector_func(
    parent_values: object,
    input: tp.Tuple[object, HighwayFlowParameters],
    activation_fn: Bijector,
    link_fn: tp.Callable[[object, object], tf.Tensor],
    bijector_fn: tp.Callable[[tf.Tensor], Bijector],
) -> Bijector:
    node_input, flow_parameters = input
    flow = HighwayFlow.from_parameters(flow_parameters, activation_fn)
    prior_params = link_fn(parent_values, node_input)
    params = flow.forward(prior_params)
    return bijector_fn(params)


def highway_flow_root_bijector_func(
    input: tp.Tuple[object, HighwayFlowParameters],
    activation_fn: Bijector,
    root_link_fn: tp.Callable[[object, object], tf.Tensor],
    bijector_fn: tp.Callable[[tf.Tensor], Bijector],
) -> Bijector:
    node_input, flow_parameters = input
    flow = HighwayFlow.from_parameters(flow_parameters, activation_fn)
    prior_params = root_link_fn(node_input)
    params = flow.forward(prior_params)
    return bijector_fn(params)


_identity_link_fn = lambda x: x


class HighwayFlowNodeBijector(PreorderNodeBijector):
    def __init__(
        self,
        topology: TensorflowTreeTopology,
        highway_flow_parameters: HighwayFlowParameters,
        node_parameter_input: object,
        activation_function: Bijector,
        link_fn: tp.Callable[[object, object], tf.Tensor] = _identity_link_fn,
        bijector_fn: tp.Callable[
            [tf.Tensor], Bijector
        ] = build_shift_and_scale_bijector,
        root_link_fn: tp.Callable[[object], tf.Tensor] = _identity_link_fn,
        name="HighwayFlowNodeBijector",
        validate_args=False,
    ):
        super().__init__(
            topology,
            (node_parameter_input, highway_flow_parameters),
            partial(
                highway_flow_bijector_func,
                activation_function=activation_function,
                link_fn=link_fn,
                bijector_fn=bijector_fn,
            ),
            partial(
                highway_flow_root_bijector_func,
                activation_function=activation_function,
                root_link_fn=root_link_fn,
                bijector_fn=bijector_fn,
            ),
            name=name,
            validate_args=validate_args,
        )
