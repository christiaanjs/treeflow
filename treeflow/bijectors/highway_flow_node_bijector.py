import typing as tp
from functools import partial, reduce
import tensorflow as tf
from tensorflow_probability.python.math import softplus_inverse
from tensorflow_probability.python.bijectors import (
    Bijector,
    Shift,
    Scale,
    Chain,
    Sigmoid,
    Identity,
)
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.bijectors.highway_flow import (
    HighwayFlow,
    HighwayFlowParameters,
    HIGHWAY_FLOW_PARAMETER_EVENT_NDIMS,
)
from treeflow.bijectors.preorder_node_bijector import PreorderNodeBijector


def build_shift_and_scale_bijector(input: tf.Tensor):
    return Chain([Shift(input[..., 1]), Scale(tf.math.softplus(input[..., 0]))])


def push_through_flow(
    params: tf.Tensor,
    index_and_activation_fn: tp.Tuple[int, Bijector],
    flow_parameters: HighwayFlowParameters,
) -> tf.Tensor:
    index, activation_fn = index_and_activation_fn
    layer_params = tf.nest.map_structure(lambda x: x[index], flow_parameters)
    flow = HighwayFlow.from_parameters(layer_params, activation_fn)
    return flow.forward(params)


def highway_flow_bijector_func(
    parent_values: object,
    input: tp.Tuple[object, HighwayFlowParameters],
    activation_functions: tp.Iterable[Bijector],
    link_fn: tp.Callable[[object, object], tf.Tensor],
    bijector_fn: tp.Callable[[tf.Tensor], Bijector],
) -> Bijector:
    node_input, flow_parameters = input

    prior_params = link_fn(parent_values, node_input)
    params = reduce(
        partial(push_through_flow, flow_parameters=flow_parameters),
        enumerate(activation_functions),
        prior_params,
    )
    return bijector_fn(params)


def highway_flow_root_bijector_func(
    input: tp.Tuple[object, HighwayFlowParameters],
    activation_functions: tp.Iterable[Bijector],
    root_link_fn: tp.Callable[[object], tf.Tensor],
    bijector_fn: tp.Callable[[tf.Tensor], Bijector],
) -> Bijector:
    node_input, flow_parameters = input
    prior_params = root_link_fn(node_input)
    params = reduce(
        partial(push_through_flow, flow_parameters=flow_parameters),
        enumerate(activation_functions),
        prior_params,
    )
    return bijector_fn(params)


_identity_link_fn = lambda x, input, n_params: tf.stack([x] * n_params, axis=-1)


def constant_root_link(input, loc, scale):
    return tf.stack([softplus_inverse(scale), loc])


DEFAULT_ACTIVATION_FUNCTIONS = (Sigmoid(), Identity())


class HighwayFlowNodeBijector(PreorderNodeBijector):
    def __init__(
        self,
        topology: TensorflowTreeTopology,
        highway_flow_parameters: HighwayFlowParameters,
        node_parameter_input: object = (),
        activation_functions: tp.Iterable[Bijector] = DEFAULT_ACTIVATION_FUNCTIONS,
        link_fn: tp.Callable[[object, object], tf.Tensor] = partial(
            _identity_link_fn, n_params=2
        ),
        bijector_fn: tp.Callable[
            [tf.Tensor], Bijector
        ] = build_shift_and_scale_bijector,
        root_link_fn: tp.Optional[tp.Callable[[object], tf.Tensor]] = None,
        init_root_loc: tp.Union[float, tf.Tensor] = 0.0,
        node_parameter_event_ndims: object = (),
        name="HighwayFlowNodeBijector",
        validate_args=False,
    ):
        dtype = highway_flow_parameters.U.dtype
        if root_link_fn is None:
            root_link_fn = partial(
                constant_root_link,
                loc=tf.convert_to_tensor(init_root_loc, dtype=dtype),
                scale=tf.constant(1.0, dtype=dtype),
            )
        super().__init__(
            topology,
            (node_parameter_input, highway_flow_parameters),
            partial(
                highway_flow_bijector_func,
                activation_functions=activation_functions,
                link_fn=link_fn,
                bijector_fn=bijector_fn,
            ),
            partial(
                highway_flow_root_bijector_func,
                activation_functions=activation_functions,
                root_link_fn=root_link_fn,
                bijector_fn=bijector_fn,
            ),
            name=name,
            input_event_ndims=(
                node_parameter_event_ndims,
                HIGHWAY_FLOW_PARAMETER_EVENT_NDIMS,
            ),
            forward_event_ndims=1,
            inverse_event_ndims=1,
            validate_args=validate_args,
        )
