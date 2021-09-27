import tensorflow as tf
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import distribution_util


def broadcast_structure(elems, event_shape, batch_shape):
    return tf.nest.map_structure(
        lambda elem, elem_event_shape: tf.broadcast_to(
            elem, tf.concat([batch_shape, elem_event_shape], 0)
        ),
        elems,
        event_shape,
    )


def reshape_structure(elems, event_shape, new_batch_shape):
    return tf.nest.map_structure(
        lambda elem, elem_event_shape: tf.reshape(
            elem, tf.concat([new_batch_shape, elem_event_shape], 0)
        ),
        elems,
        event_shape,
    )


def vectorize_over_batch_dims(
    fn, elems, event_shape, batch_shape, vectorized_map=True, fn_output_signature=None
):
    flat_batch_shape = tf.expand_dims(ps.reduce_prod(batch_shape), 0)
    flat_structure = reshape_structure(elems, event_shape, flat_batch_shape)
    if vectorized_map:
        result = tf.vectorized_map(fn, flat_structure, fallback_to_while_loop=False)
    else:
        assert fn_output_signature is not None
        result = tf.map_fn(fn, flat_structure, fn_output_signature=fn_output_signature)
    new_event_shape = tf.nest.map_structure(lambda elem: tf.shape(elem)[1:], result)
    return reshape_structure(result, new_event_shape, batch_shape)
