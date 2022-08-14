from functools import partial
import typing as tp
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.bijectors as tfb
from tensorflow_probability.python.experimental.vi.util import (
    build_trainable_linear_operator_block,
)
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.experimental.util import make_trainable
import tensorflow.python.util.nest as tf_nest
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


def get_mean_field_approximation(
    model: tfd.JointDistribution,
    init_loc=None,
    dtype=DEFAULT_FLOAT_DTYPE_TF,
    joint_bijector_func: tp.Callable[
        [tfd.JointDistribution], tfb.Composition
    ] = get_default_event_space_bijector,
    event_shape_fn: tp.Callable[[tfd.JointDistribution], object] = event_shape_fn,
) -> tfd.Distribution:
    # operator_classes = get_mean_field_operator_classes(flat_event_size)
    # linear_operator_block = build_trainable_linear_operator_block(
    #     operator_classes, flat_event_size, dtype=dtype
    # )
    # scale_bijector = tfb.ScaleMatvecLinearOperatorBlock(linear_operator_block)

    # loc_bijector = get_trainable_shift_bijector(
    #     flat_event_size, init_loc_1d, dtype=dtype
    # )

    # base_standard_dist = get_base_distribution(flat_event_size, dtype=dtype)
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
    base_standard_dist = tfd.JointDistributionSequential(
        tf.nest.map_structure(
            lambda s, init: tfd.Independent(
                make_trainable(
                    tfd.Normal,
                    initial_parameters=(None if init is None else dict(loc=init)),
                    batch_and_event_shape=s,
                    parameter_dtype=dtype,
                ),
                reinterpreted_batch_ndims=tensorshape_util.rank(s),
            ),
            base_event_shape,
            init_loc_1d,
        )
    )

    distribution = tfd.TransformedDistribution(
        base_standard_dist, event_shape_and_space_bijector
    )
    return distribution


def get_fixed_topology_mean_field_approximation(
    model: tfd.JointDistribution,
    topology_pins: tp.Dict[str, TensorflowTreeTopology],
    init_loc=None,
    dtype=DEFAULT_FLOAT_DTYPE_TF,
) -> tfd.Distribution:
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
