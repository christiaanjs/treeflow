from functools import partial
import typing as tp
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.bijectors as tfb
from tensorflow_probability.python.internal import tensorshape_util
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.model.event_shape_bijector import (
    event_shape_fn,
    get_default_event_space_bijector,
    get_unconstrained_init_values,
    inverse_with_nones,
    get_event_shape_and_space_bijector,
    get_fixed_topology_joint_bijector,
    get_fixed_topology_event_shape,
)
from tensorflow_probability.python.experimental.util import (
    deferred_module,
)
from tensorflow_probability.python.experimental.util.trainable import _make_trainable
from tensorflow_probability.python.internal.trainable_state_util import (
    _initialize_parameters,
    _apply_parameters,
)


def get_base_distribution(flat_event_size, dtype=DEFAULT_FLOAT_DTYPE_TF):
    base_standard_dist = tfd.JointDistributionSequential(
        [
            tfd.Sample(
                tfd.Normal(
                    loc=tf.constant(0.0, dtype=dtype),
                    scale=tf.constant(1.0, dtype=dtype),
                ),
                s,
            )
            for s in flat_event_size
        ]
    )
    return base_standard_dist


def get_mean_field_operator_classes(flat_event_size):
    return tuple(
        [tf.linalg.LinearOperatorDiag for _ in flat_event_size]
    )  # TODO: Bijector?


def get_trainable_shift_bijector(
    flat_event_size, init_loc_unconstrained, dtype=DEFAULT_FLOAT_DTYPE_TF
):
    return tfb.JointMap(
        tf.nest.map_structure(
            lambda s, init: tfb.Shift(
                tf.Variable(
                    tf.random.uniform((s,), minval=-2.0, maxval=2.0, dtype=dtype)
                    if init is None
                    else init
                )
            ),
            flat_event_size,
            init_loc_unconstrained,
        )
    )


def make_trainable(
    dist_class: tp.Type[tfd.Distribution],
    initial_parameters: tp.Optional[tp.Dict[str, object]] = None,
    batch_and_event_shape: tf.Tensor = None,
    parameter_dtype: tf.DType = DEFAULT_FLOAT_DTYPE_TF,
    seed=None,
    var_name_prefix="",
    **init_kwargs,
) -> tfd.Distribution:
    g = partial(
        _make_trainable,
        cls=dist_class,
        initial_parameters=initial_parameters,
        batch_and_event_shape=batch_and_event_shape,
        parameter_dtype=parameter_dtype,
        **init_kwargs,
    )
    params = _initialize_parameters(g, seed=seed)
    params_as_variables = []
    for name, value in params._asdict().items():
        # Params may themselves be structures, in which case there's no 1:1
        # mapping between param names and variable names. Currently we just give
        # the same name to all variables in a param structure and let TF sort
        # things out.
        params_as_variables.append(
            tf.nest.map_structure(
                lambda t, n=name: t
                if t is None
                else tf.Variable(t, name=var_name_prefix + n),
                value,
                expand_composites=True,
            )
        )
    return deferred_module.DeferredModule(
        partial(_apply_parameters, g),
        *params_as_variables,
        also_track=tf.nest.flatten(
            (initial_parameters, init_kwargs)
        ),  # TODO: Could other args be trainable variables?
    )


def get_mean_field_approximation(
    model: tfd.JointDistribution,
    init_loc=None,
    dtype=DEFAULT_FLOAT_DTYPE_TF,
    joint_bijector_func: tp.Callable[
        [tfd.JointDistribution], tfb.Composition
    ] = get_default_event_space_bijector,
    event_shape_fn: tp.Callable[[tfd.JointDistribution], object] = event_shape_fn,
    seed=None,
) -> tp.Tuple[tfd.Distribution, tp.Dict[str, tf.Variable]]:
    (
        event_shape_and_space_bijector,
        base_event_shape,
    ) = get_event_shape_and_space_bijector(
        model, joint_bijector_func=joint_bijector_func, event_shape_fn=event_shape_fn
    )
    init_loc_1d = get_unconstrained_init_values(
        model,
        event_shape_and_space_bijector,
        event_shape_fn=event_shape_fn,
        init=init_loc,
    )
    base_standard_dist = tfd.JointDistributionNamed(
        {
            name: tfd.Independent(
                make_trainable(
                    tfd.Normal,
                    initial_parameters=(
                        None
                        if init_loc_1d[name] is None
                        else dict(loc=init_loc_1d[name])
                    ),
                    batch_and_event_shape=s,
                    parameter_dtype=dtype,
                    seed=seed,
                    var_name_prefix=name + "_",
                ),
                reinterpreted_batch_ndims=tensorshape_util.rank(s),
            )
            for name, s in base_event_shape.items()
        }
    )

    distribution = tfd.TransformedDistribution(
        base_standard_dist, event_shape_and_space_bijector
    )
    variables_dict = {
        variable.name: variable for variable in distribution.trainable_variables
    }
    return distribution, variables_dict


def get_fixed_topology_mean_field_approximation(
    model: tfd.JointDistribution,
    topology_pins: tp.Dict[str, TensorflowTreeTopology],
    init_loc=None,
    dtype=DEFAULT_FLOAT_DTYPE_TF,
) -> tp.Tuple[tfd.Distribution, tp.Dict[str, tf.Tensor]]:
    bijector_func = partial(
        get_fixed_topology_joint_bijector, topology_pins=topology_pins
    )
    event_shape_fn = partial(
        get_fixed_topology_event_shape, topology_pins=topology_pins
    )
    return get_mean_field_approximation(
        model,
        init_loc=init_loc,
        dtype=dtype,
        joint_bijector_func=bijector_func,
        event_shape_fn=event_shape_fn,
    )
