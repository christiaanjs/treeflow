from __future__ import annotations

import attr
import tensorflow as tf
import tensorflow.python.util.nest as nest


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

    def add_inner_batch_dimensions(self, batch_dims: int = 1) -> Eigendecomposition:
        """
        Add batch dimensions before the state dimensions
        """
        # TODO: Make this work with dynamic `batch_dims`
        assert batch_dims >= 0
        if batch_dims > 0:
            return nest.map_structure(
                lambda x, dim: tf.expand_dims(x, axis=dim),
                self.add_inner_batch_dimensions(batch_dims - 1),
                Eigendecomposition(
                    eigenvectors=-3, inverse_eigenvectors=-3, eigenvalues=-2
                ),
            )
        else:
            return self


__all__ = [Eigendecomposition.__name__]
