import tensorflow as tf

def _vectorize_1d(f, x):
    batch_shape = x.shape[:-1]
    batch_size = tf.reduce_prod(batch_shape)
    elems = tf.reshape(x, [batch_size, x.shape[-1]])
    res_flat = tf.map_fn(f, elems)
    return tf.reshape(res_flat, batch_shape + x.shape[-1])

def vectorize_1d_if_needed(f, x):
    return tf.cond(x.shape.rank == 1,
        lambda: f(x),
        lambda: _vectorize_1d(f, x)
    )