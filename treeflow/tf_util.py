import tensorflow as tf

def get_nested_shape(x):
    return x.shape if isinstance(x, tf.Tensor) else x[0].shape

def reshape_batch_dims(x, batch_ndims, new_batch_shape):
    return tf.reshape(x, new_batch_shape + x.shape[batch_ndims:].as_list())

def apply_nested(f, x):
    return f(x) if isinstance(x, tf.Tensor) else [f(y) for y in x]

def _vectorize_1d(f, x, batch_ndims, dtype):
    shape = get_nested_shape(x)
    batch_shape = shape[:batch_ndims]
    batch_size = tf.reduce_prod(batch_shape)
    elems = apply_nested(lambda y: reshape_batch_dims(y, batch_ndims, [batch_size]), x)
    res_flat = tf.map_fn(f, elems, dtype=dtype)
    return apply_nested(lambda y: reshape_batch_dims(y, batch_ndims, batch_shape), res_flat)

def vectorize_1d_if_needed(f, x, batch_ndims, dtype=tf.float32):
    return f(x) if batch_ndims == 0 else _vectorize_1d(f, x, batch_ndims, dtype)