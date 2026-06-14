import tensorflow as tf

from treeflow.evolution.substitution.base_substitution_model import (
    EigendecompositionSubstitutionModel,
)


class DiscreteTraitModel(EigendecompositionSubstitutionModel):
    """Time-reversible K-state discrete-trait substitution model.

    Follows the parameterisation of Lemey et al. (2009) for Bayesian
    phylogeography: Q[i,j] = rate[pair(i,j)] * pi[j] for i != j, and the
    diagonal is set to ensure zero row sums.

    The rate vector carries the K*(K-1)/2 symmetric exchange rates in
    row-major upper-triangular order: (0,1), (0,2), ..., (0,K-1),
    (1,2), (1,3), ..., (K-2, K-1).
    """

    def __init__(self, n_states: int):
        if n_states < 2:
            raise ValueError(f"n_states must be >= 2, got {n_states}")
        self.n_states = int(n_states)
        self._upper_indices = _upper_triangle_indices(self.n_states)

    def q(self, frequencies: tf.Tensor, rates: tf.Tensor) -> tf.Tensor:
        K = self.n_states
        expected_rates = K * (K - 1) // 2
        rates = tf.convert_to_tensor(rates)
        frequencies = tf.convert_to_tensor(frequencies)

        tf.debugging.assert_equal(
            tf.shape(rates)[-1],
            expected_rates,
            message=(
                f"DiscreteTraitModel({K}) expects rates of length "
                f"{expected_rates} (K*(K-1)/2)."
            ),
        )
        tf.debugging.assert_equal(
            tf.shape(frequencies)[-1],
            K,
            message=f"DiscreteTraitModel({K}) expects frequencies of length {K}.",
        )

        exchange = _scatter_symmetric(rates, K, self._upper_indices)
        pi_row = frequencies[..., tf.newaxis, :]

        off_diag = exchange * pi_row
        row_sum = tf.reduce_sum(off_diag, axis=-1)
        q = tf.linalg.set_diag(off_diag, -row_sum)
        return q


def _upper_triangle_indices(K: int) -> tf.Tensor:
    """(K*(K-1)/2, 2) int32 tensor of (i, j) pairs with i < j, row-major."""
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    return tf.constant(pairs, dtype=tf.int32)


def _scatter_symmetric(
    rates: tf.Tensor, K: int, upper_indices: tf.Tensor
) -> tf.Tensor:
    """Build a (..., K, K) symmetric matrix with zero diagonal from a
    (..., K*(K-1)/2) row-major upper-triangular rate vector."""
    batch_shape = tf.shape(rates)[:-1]
    flat_len = K * (K - 1) // 2

    # Flatten the batch, scatter per-batch, then reshape back.
    rates_flat = tf.reshape(rates, [-1, flat_len])
    batch_size = tf.shape(rates_flat)[0]

    def _build_one(rates_1d):
        upper = tf.scatter_nd(upper_indices, rates_1d, shape=(K, K))
        return upper + tf.linalg.matrix_transpose(upper)

    out = tf.map_fn(_build_one, rates_flat)
    return tf.reshape(out, tf.concat([batch_shape, [K, K]], axis=0))


__all__ = ["DiscreteTraitModel"]
