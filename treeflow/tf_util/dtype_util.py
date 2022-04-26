import tensorflow as tf
import numpy as np

DEFAULT_FLOAT_DTYPE_TF = tf.float64
DEFAULT_FLOAT_DTYPE_NP = np.float64


def float_constant(x):
    return tf.constant(x, dtype=DEFAULT_FLOAT_DTYPE_TF)


__all__ = ["DEFAULT_FLOAT_DTYPE_TF", "DEFAULT_FLOAT_DTYPE_NP", "float_constant"]
