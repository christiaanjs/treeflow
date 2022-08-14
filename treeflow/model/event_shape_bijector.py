import typing as tp

import tensorflow as tf
import tensorflow.python.util.nest as tf_nest
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.bijectors as tfb
from tensorflow_probability.python.internal import tensorshape_util


def inverse_with_nones(bijector: tfb.Bijector, val):
    if isinstance(bijector, tfb.Composition):
        return bijector._walk_inverse(inverse_with_nones, val)
    if val is None:
        return None
    else:
        return bijector.inverse(val)


# def joint_inverse_with_nones(bijector: tfb.Composition, val):
#     return bijector._walk_inverse(inverse_with_nones, val)


def get_default_event_space_bijector(model: tfd.JointDistribution) -> tfb.Composition:
    return model.experimental_default_event_space_bijector()


def event_shape_fn(model: tfd.JointDistribution):
    return model.event_shape_tensor()


def get_event_shape_bijector(
    model: tfd.JointDistribution,
    joint_bijector_func: tp.Callable[
        [tfd.JointDistribution], tfb.Composition
    ] = get_default_event_space_bijector,
    event_shape_fn: tp.Callable[[tfd.JointDistribution], object] = event_shape_fn,
) -> tfd.Distribution:
    event_space_bijector = joint_bijector_func(model)
    event_shape = event_shape_fn(model)
    flat_event_shape = model._model_flatten(event_shape)
    flat_model_event_shape = model._model_flatten(model.event_shape)
    # Some bijectors (e.g. SoftmaxCentered) change event shape, but we need to handle trees
    unconstrained_event_shape = [
        (
            bijector.inverse_event_shape(shape)
            if isinstance(model_shape, tf.TensorShape)
            else shape
        )
        for bijector, shape, model_shape in zip(
            event_space_bijector.bijectors, flat_event_shape, flat_model_event_shape
        )
    ]
    flat_event_size = tf_nest.map_structure(tf.reduce_prod, unconstrained_event_shape)

    unflatten_bijector = tfb.Restructure(
        model._model_unflatten(range(len(unconstrained_event_shape)))
    )
    reshape_bijector = tfb.JointMap(
        tf_nest.map_structure(
            lambda event_shape_element, event_size_element: tfb.Reshape(
                event_shape_element, tf.expand_dims(event_size_element, 0)
            )
            if tensorshape_util.rank(event_shape_element) > 0
            else tfb.Identity(),
            unconstrained_event_shape,
            flat_event_size,
        )
    )
    base_event_shape = reshape_bijector.inverse_event_shape(unconstrained_event_shape)
    chain_bijector = tfb.Chain(
        [
            event_space_bijector,
            unflatten_bijector,
            reshape_bijector,
        ]
    )
    return chain_bijector, base_event_shape
