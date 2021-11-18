import tensorflow as tf
import treeflow


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


obs = tf.convert_to_tensor([0.8, 1.2], dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF)


def obj(vars):
    return lambda: tf.math.square(vars["a"]) + tf.math.reduce_sum(
        tf.math.square(vars["a"] - vars["b"]) + tf.math.square(vars["b"] - obs)
    )


def optimizer_builder():
    return tf.optimizers.SGD(learning_rate=1e-2)


__all__ = [vars.__name__, obj.__name__, optimizer_builder.__name__]
