import attr
import tensorflow as tf


@attr.s(auto_attribs=True, slots=True)
class Eigendecomposition:
    """
    Eigendecomposition of an instantaneous rate matrix

    Attributes
    ----------
    eigenvectors
        2D Tensor with right eigenvectors as columns
    inverse_eigenvectors
        2D Tensor, inverse of `eigenvectors`
    eigenvalues
        1D Tensor of eigenvalues
    """

    eigenvectors: tf.Tensor
    inverse_eigenvectors: tf.Tensor
    eigenvalues: tf.Tensor


__all__ = [Eigendecomposition.__name__]
