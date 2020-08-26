import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from functools import reduce

def get_nested_shape(x):
    return tf.shape(x) if isinstance(x, tf.Tensor) else tf.shape(x[0])

def reshape_batch_dims(x, batch_ndims, new_batch_shape):
    return tf.reshape(x, tf.concat([new_batch_shape, tf.shape(x)[batch_ndims:]], axis=0))

def apply_nested(f, x):
    return f(x) if isinstance(x, tf.Tensor) else [f(y) for y in x]

def prod(x):
    return reduce(lambda a, b: a * b, x)

def _vectorize_1d(f, x, batch_ndims, dtype):
    shape = get_nested_shape(x)
    batch_shape = shape[:batch_ndims]
    batch_size = tf.reduce_prod(batch_shape)
    elems = apply_nested(lambda y: reshape_batch_dims(y, batch_ndims, [batch_size]), x)
    res_flat = tf.map_fn(f, elems, fn_output_signature=dtype)
    return apply_nested(lambda y: reshape_batch_dims(y, batch_ndims, batch_shape), res_flat)

def vectorize_1d_if_needed(f, x, batch_ndims, dtype=DEFAULT_FLOAT_DTYPE_TF):
    return tf.cond(batch_ndims == 0, lambda: f(x), lambda: _vectorize_1d(f, x, batch_ndims, dtype))
