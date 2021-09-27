import pytest
import attr
import tensorflow as tf
from treeflow.tf_util.vectorize import broadcast_structure, vectorize_over_batch_dims
from numpy.testing import assert_allclose


@attr.attrs(auto_attribs=True)
class InnerContainer:
    a: tf.Tensor
    b: tf.Tensor


@attr.attrs(auto_attribs=True)
class Container:
    inner: InnerContainer
    c: tf.Tensor


def test_broadcast_structure():
    a = tf.reshape(tf.range(9), [1, 3, 3])
    b = tf.reshape(tf.range(18), [3, 2, 3])
    c = tf.reshape(tf.range(2), [2, 1])

    structure = Container(inner=InnerContainer(a=a, b=b), c=c)
    event_shape = Container(
        inner=InnerContainer(
            a=tf.convert_to_tensor([3]), b=tf.convert_to_tensor([2, 3])
        ),
        c=tf.convert_to_tensor((), dtype=tf.int32),
    )

    batch_shape = (2, 3)
    res = broadcast_structure(structure, event_shape, batch_shape)
    assert res.inner.a.shape == (2, 3, 3)
    assert res.inner.b.shape == (2, 3, 2, 3)
    assert res.c.shape == (2, 3)


@pytest.mark.parametrize("function_mode", [False, True])
@pytest.mark.parametrize("vectorized_map", [False, True])
def test_vectorize_over_batch_dims_scalar(vectorized_map, function_mode):

    a = tf.reshape(tf.range(36), [3, 2, 3, 2])
    b = tf.reshape(tf.range(36, 72), [3, 2, 2, 3])
    c = tf.reshape(tf.range(72, 90), [3, 2, 3])

    def func(container):
        return tf.reduce_sum(
            tf.matmul(container.inner.a, container.inner.b) * container.c
        )

    structure = Container(inner=InnerContainer(a=a, b=b), c=c)
    event_shape = tf.nest.map_structure(lambda x: tf.shape(x)[2:], structure)
    batch_shape = [3, 2]

    def outer_func(arg):
        return vectorize_over_batch_dims(
            func,
            structure,
            event_shape,
            batch_shape,
            vectorized_map=vectorized_map,
            fn_output_signature=tf.int32,
        )

    if function_mode:
        outer_func = tf.function(outer_func)

    res = outer_func(structure)

    inner_expected = tf.reduce_sum(
        tf.expand_dims(a, -1) * tf.expand_dims(b, -3), axis=-2
    )
    expected = tf.reduce_sum(
        tf.reduce_sum(inner_expected * tf.expand_dims(c, -2), axis=-1), axis=-1
    )
    assert_allclose(res.numpy(), expected.numpy())


def test_vectorize_over_batch_dims_structure():
    pass  # TODO
