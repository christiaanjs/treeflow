import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from functools import reduce


def get_nested_shape(x):
    return tf.shape(x) if isinstance(x, tf.Tensor) else tf.shape(x[0])


def flatten_batch_dims(x, batch_ndims, batch_size):
    new_shape = tf.concat([[batch_size], tf.shape(x)[batch_ndims:]], axis=0)
    return tf.reshape(x, new_shape)


def unflatten_batch_dims(x, batch_ndims, batch_shape):
    event_shape = tf.shape(x)[1:]
    new_shape = tf.concat([batch_shape, event_shape], axis=0)
    return tf.reshape(x, new_shape)


def apply_nested(f, x):
    return f(x) if isinstance(x, tf.Tensor) else [f(y) for y in x]


def prod(x):
    return reduce(lambda a, b: a * b, x)


def vectorize(f, x, batch_ndims, dtype):
    shape = get_nested_shape(x)
    batch_shape = shape[:batch_ndims]
    batch_size = tf.reduce_prod(batch_shape)
    elems = apply_nested(lambda y: flatten_batch_dims(y, batch_ndims, batch_size), x)
    res_flat = tf.map_fn(f, elems, fn_output_signature=dtype)
    return apply_nested(
        lambda y: unflatten_batch_dims(y, batch_ndims, batch_shape), res_flat
    )


def vectorize_1d_if_needed(f, x, batch_ndims, dtype=DEFAULT_FLOAT_DTYPE_TF):
    return vectorize(f, x, batch_ndims, dtype)
