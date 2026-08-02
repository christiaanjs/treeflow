from functools import partial
import typing as tp
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.bijectors as tfb
from tensorflow_probability.python.util import DeferredTensor
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.model.event_shape_bijector import (
    event_shape_fn,
    get_default_event_space_bijector,
    get_event_shape_and_space_bijector,
    get_fixed_topology_joint_bijector,
    get_fixed_topology_event_shape,
    get_unconstrained_init_values,
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
    init_loc=None,
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
    total_dim = tf.reduce_sum(flat_event_size_tensor)

    # Trainable affine base: z → softplus(log_scale) * z + loc.
    # At network init (zero outputs) the IAF is the identity bijector, so the
    # distribution is Normal(loc, softplus(log_scale)) in unconstrained space —
    # identical to a mean-field approximation initialised at init_loc.
    init_loc_1d = get_unconstrained_init_values(
        model,
        event_shape_and_space_bijector,
        event_shape_fn=event_shape_fn,
        init=init_loc,
    )
    # tf.nest.flatten on a dict sorts keys alphabetically, matching the
    # ordering produced by restructure_bijector.inverse_event_shape.
    flat_init_locs = tf.nest.flatten(init_loc_1d)
    flat_loc_init = tf.concat(
        [
            x if x is not None else tf.zeros(s, dtype=dtype)
            for x, s in zip(flat_init_locs, flat_event_size)
        ],
        axis=0,
    )
    loc_var = tf.Variable(flat_loc_init, name="iaf_base_loc")
    # Initialise scale to 1.0 (softplus_inverse(1) = log(e - 1) ≈ 0.541)
    softplus_inv1 = tf.cast(
        tf.math.log(tf.exp(tf.ones([], dtype=tf.float32)) - 1.0), dtype
    )
    log_scale_var = tf.Variable(
        tf.fill([total_dim], softplus_inv1), name="iaf_base_log_scale"
    )

    base_dist = tfd.Sample(
        tfd.Normal(tf.constant(0.0, dtype=dtype), tf.constant(1.0, dtype=dtype)),
        total_dim,
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
                    # Small init so the IAF is near-identity at the start;
                    # avoids extreme samples that make the warm-up loss NaN.
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
                )
            )
        )
        for _ in range(n_iaf_bijectors)
    ]
    # DeferredTensor evaluates softplus lazily inside each graph call so that
    # gradients flow back to log_scale_var.  A plain tf.math.softplus(var)
    # produces a constant EagerTensor at bijector construction time.
    chain_bijector = tfb.Chain(
        [
            event_shape_and_space_bijector,
            restructure_bijector,
            split_bijector,
        ]
        + iaf_bijectors
        + [
            tfb.Shift(loc_var),
            tfb.Scale(DeferredTensor(log_scale_var, tf.math.softplus)),
        ]
    )
    distribution = tfd.TransformedDistribution(base_dist, chain_bijector)
    distribution.sample()  # Build networks
    network_weights = [
        weight_variable
        for bijector in iaf_bijectors
        for weight_variable in bijector.bijector._shift_and_log_scale_fn._network.weights
    ]
    variables = network_weights + [loc_var, log_scale_var]
    variables_dict = {variable.name: variable for variable in variables}
    return distribution, variables_dict


def get_fixed_topology_inverse_autoregressive_flow_approximation(
    model: tfd.JointDistribution,
    hidden_units_per_layer: int,
    topology_pins: tp.Dict[str, TensorflowTreeTopology],
    dtype=DEFAULT_FLOAT_DTYPE_TF,
    n_hidden_layers: int = DEFAULT_N_HIDDEN_LAYERS,
    n_iaf_bijectors: int = DEFAULT_N_IAF_BIJECTORS,
    init_loc=None,
    use_native="auto",
    unroll="auto",
) -> tp.Tuple[tfd.Distribution, tp.Dict[str, tf.Tensor]]:
    bijector_func = partial(
        get_fixed_topology_joint_bijector,
        topology_pins=topology_pins,
        use_native=use_native,
        unroll=unroll,
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
        init_loc=init_loc,
    )
