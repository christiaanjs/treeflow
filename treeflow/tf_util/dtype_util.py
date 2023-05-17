import typing as tp
import tensorflow as tf
import numpy as np

DEFAULT_FLOAT_DTYPE_TF = tf.float64
DEFAULT_FLOAT_DTYPE_NP = np.float64


def float_constant(x: tp.Union[float, np.ndarray, tp.Iterable[float]]):
    """
    Converts a floating point value or array to a constant Tensor with TreeFlow's
    default data type

    Parameters
    ----------
    x
        Value that can be converted to a Tensor

    Returns
    -------
    tf.Tensor
        Value converted to a constant tensor
    """
    return tf.constant(x, dtype=DEFAULT_FLOAT_DTYPE_TF)


__all__ = ["DEFAULT_FLOAT_DTYPE_TF", "DEFAULT_FLOAT_DTYPE_NP", "float_constant"]
