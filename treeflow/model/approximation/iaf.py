from functools import partial
import typing as tp
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.bijectors as tfb
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.model.event_shape_bijector import (
    event_shape_fn,
    get_default_event_space_bijector,
    get_event_shape_and_space_bijector,
    get_fixed_topology_joint_bijector,
    get_fixed_topology_event_shape,
)

DEFAULT_N_HIDDEN_LAYERS = 2
DEFAULT_N_IAF_BIJECTORS = 2


def get_inverse_autoregressive_flow_approximation(
    model: tfd.JointDistribution,
    hidden_units_per_layer: int,
    dtype=DEFAULT_FLOAT_DTYPE_TF,
    joint_bijector_func: tp.Callable[
        [tfd.JointDistribution], tfb.Composition
    ] = get_default_event_space_bijector,
    event_shape_fn: tp.Callable[[tfd.JointDistribution], object] = event_shape_fn,
    seed=None,
    n_hidden_layers: int = DEFAULT_N_HIDDEN_LAYERS,
    n_iaf_bijectors: int = DEFAULT_N_IAF_BIJECTORS,
) -> tp.Tuple[tfd.Distribution, tp.Dict[str, tf.Variable]]:
    (
        event_shape_and_space_bijector,
        base_event_shape,
    ) = get_event_shape_and_space_bijector(
        model, joint_bijector_func=joint_bijector_func, event_shape_fn=event_shape_fn
    )
    restructure_bijector = tfb.Restructure(
        output_structure=tf.nest.pack_sequence_as(
            base_event_shape, range(len(base_event_shape))
        ),
    )
    flat_event_size = restructure_bijector.inverse_event_shape(base_event_shape)
    flat_event_size_tensor = tf.concat(flat_event_size, axis=0)
    split_bijector = tfb.Split(flat_event_size_tensor)
    base_event_shape = tf.reduce_sum(flat_event_size_tensor)
    base_dist = tfd.Sample(
        tfd.Normal(tf.constant(0.0, dtype=dtype), tf.constant(1.0, dtype=dtype)),
        base_event_shape,
    )
    iaf_bijectors = [
        tfb.Invert(
            tfb.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
                    params=2,
                    hidden_units=n_hidden_layers * [hidden_units_per_layer],
                    activation="relu",
                    dtype=dtype,
                    event_shape=base_dist.event_shape,
                )
            )
        )
        for _ in range(n_iaf_bijectors)
    ]
    chain_bijector = tfb.Chain(
        [
            event_shape_and_space_bijector,
            restructure_bijector,
            split_bijector,
        ]
        + iaf_bijectors
    )
    distribution = tfd.TransformedDistribution(base_dist, chain_bijector)
    distribution.sample()  # Build networks
    variables = [
        weight_variable
        for bijector in iaf_bijectors
        for weight_variable in bijector.bijector._shift_and_log_scale_fn._network.weights
    ]
    variables_dict = {variable.name: variable for variable in variables}
    return distribution, variables_dict


def get_fixed_topology_inverse_autoregressive_flow_approximation(
    model: tfd.JointDistribution,
    hidden_units_per_layer: int,
    topology_pins: tp.Dict[str, TensorflowTreeTopology],
    dtype=DEFAULT_FLOAT_DTYPE_TF,
    n_hidden_layers: int = DEFAULT_N_HIDDEN_LAYERS,
    n_iaf_bijectors: int = DEFAULT_N_IAF_BIJECTORS,
    init_loc=None,  # ignored
) -> tp.Tuple[tfd.Distribution, tp.Dict[str, tf.Tensor]]:
    bijector_func = partial(
        get_fixed_topology_joint_bijector, topology_pins=topology_pins
    )
    event_shape_fn = partial(
        get_fixed_topology_event_shape, topology_pins=topology_pins
    )
    return get_inverse_autoregressive_flow_approximation(
        model,
        hidden_units_per_layer,
        dtype=dtype,
        joint_bijector_func=bijector_func,
        event_shape_fn=event_shape_fn,
        n_hidden_layers=n_hidden_layers,
        n_iaf_bijectors=n_iaf_bijectors,
    )
