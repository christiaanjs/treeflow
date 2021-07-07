import tensorflow as tf
import pytest


def vars(prefix=""):
    return dict(
        a=tf.Variable(
            tf.convert_to_tensor(1.0, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF),
            name=f"{prefix}a",
        ),
        b=tf.Variable(
            tf.convert_to_tensor([1.2, 3.2], dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF),
            name=f"{prefix}b",
        ),
    )


def obj():
    def _obj(vars):
        return lambda: tf.math.square(vars["a"]) + tf.math.reduce_sum(
            tf.math.square(vars["a"] - vars["b"])
        )

    return obj


def optimizer_builder():
    return tf.optimizers.SGD(learning_rate=1e-2)


def test_minimize_eager_matches():
    optimizer = optimizer_builder()
    other_optimizer = optimizer_builder()

    _vars = vars()
    other_vars = vars("other_")
    assert False
