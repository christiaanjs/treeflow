from collections import defaultdict
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.bijectors as tfb
from tensorflow_probability.python.experimental.vi.util import (
    build_trainable_linear_operator_block,
)
from torch import distributions
from treeflow import DEFAULT_FLOAT_DTYPE_TF


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


def inverse_with_nones(bijector: tfb.Composition, val):
    return bijector._walk_inverse(
        lambda bij, element: element if element is None else bij.inverse(element), val
    )


def get_mean_field_approximation(
    model: tfd.JointDistribution, init_loc=None, dtype=DEFAULT_FLOAT_DTYPE_TF
):
    event_shape = model.event_shape_tensor()
    flat_event_shape = tf.nest.flatten(event_shape)
    flat_event_size = tf.nest.map_structure(tf.reduce_prod, flat_event_shape)
    operator_classes = get_mean_field_operator_classes(flat_event_size)
    linear_operator_block = build_trainable_linear_operator_block(
        operator_classes, flat_event_size, dtype=dtype
    )
    scale_bijector = tfb.ScaleMatvecLinearOperatorBlock(linear_operator_block)

    if init_loc is None:
        init_loc = tf.nest.map_structure(lambda _: None, flat_event_shape)
    else:
        init_loc = defaultdict(lambda: None, init_loc)  # TODO: Handle nesting

    event_space_bijector: tfb.Composition = (
        model.experimental_default_event_space_bijector()
    )
    unflatten_bijector = tfb.Restructure(
        tf.nest.pack_sequence_as(event_shape, range(len(flat_event_shape)))
    )
    reshape_bijector = tfb.JointMap(
        tf.nest.map_structure(tfb.Reshape, flat_event_shape)
    )

    init_loc_unconstrained = inverse_with_nones(event_space_bijector, init_loc)
    init_loc_flat = unflatten_bijector.inverse(init_loc_unconstrained)
    init_loc_1d = inverse_with_nones(reshape_bijector, init_loc_flat)
    loc_bijector = get_trainable_shift_bijector(
        flat_event_size, init_loc_1d, dtype=dtype
    )

    base_standard_dist = get_base_distribution(flat_event_size, dtype=dtype)
    chain_bijector = tfb.Chain(
        [
            event_space_bijector,
            unflatten_bijector,
            reshape_bijector,
            loc_bijector,
            scale_bijector,
        ]
    )
    distribution = tfd.TransformedDistribution(base_standard_dist, chain_bijector)
    return distribution
