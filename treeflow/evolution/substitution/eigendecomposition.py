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

    def add_inner_batch_dimensions(
        self,
        batch_dims: int = 1,
        inner_batch_rank: int = 0,
    ) -> Eigendecomposition:
        """
        Add batch dimensions before the state dimensions
        """
        # TODO: Reimplement with reshape
        assert batch_dims >= 0
        if batch_dims > 0:
            return nest.map_structure(
                lambda x, dim: tf.expand_dims(x, axis=dim),
                self.add_inner_batch_dimensions(
                    batch_dims - 1, inner_batch_rank=inner_batch_rank
                ),
                Eigendecomposition(
                    eigenvectors=-3 - inner_batch_rank,
                    inverse_eigenvectors=-3 - inner_batch_rank,
                    eigenvalues=-2 - inner_batch_rank,
                ),
            )
        else:
            return self


__all__ = [Eigendecomposition.__name__]
