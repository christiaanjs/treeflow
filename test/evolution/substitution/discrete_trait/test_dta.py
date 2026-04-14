import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

from treeflow.evolution.substitution.discrete_trait.dta import DiscreteTraitModel
from treeflow.evolution.substitution.nucleotide.jc import JC


@pytest.mark.parametrize("n_states", [2, 3, 4, 5, 8])
def test_q_shape_and_zero_row_sums(tensor_constant, n_states):
    model = DiscreteTraitModel(n_states)
    n_rates = n_states * (n_states - 1) // 2
    frequencies = tensor_constant(np.full(n_states, 1.0 / n_states))
    rates = tensor_constant(np.linspace(0.5, 1.5, n_rates))

    q = model.q(frequencies=frequencies, rates=rates)

    assert q.shape == (n_states, n_states)
    row_sums = tf.reduce_sum(q, axis=-1).numpy()
    assert_allclose(row_sums, np.zeros(n_states), atol=1e-10)


def test_time_reversibility(tensor_constant):
    # pi_i Q_ij == pi_j Q_ji for all i != j
    K = 5
    model = DiscreteTraitModel(K)
    pi = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    rates = np.linspace(0.3, 1.7, K * (K - 1) // 2)
    q = model.q(
        frequencies=tensor_constant(pi), rates=tensor_constant(rates)
    ).numpy()

    lhs = pi[:, None] * q
    # lhs[i, j] = pi_i * Q_ij should equal lhs[j, i] = pi_j * Q_ji
    assert_allclose(lhs, lhs.T, atol=1e-12)


def test_reduces_to_jc_when_k4_equal_rates(tensor_constant):
    """With K=4, equal frequencies, and equal rates, Q_norm should match JC."""
    K = 4
    model = DiscreteTraitModel(K)
    pi = tensor_constant(np.full(K, 1.0 / K))
    rates = tensor_constant(np.ones(K * (K - 1) // 2))

    q_dta = model.q_norm(frequencies=pi, rates=rates).numpy()
    q_jc = JC().q_norm(frequencies=pi).numpy()
    assert_allclose(q_dta, q_jc, atol=1e-12)


def test_eigendecomposition_roundtrip(tensor_constant):
    K = 6
    model = DiscreteTraitModel(K)
    pi = tensor_constant(np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.25]))
    rates = tensor_constant(np.linspace(0.2, 2.0, K * (K - 1) // 2))

    eig = model.eigen(frequencies=pi, rates=rates)
    q_reconstructed = (
        eig.eigenvectors @ tf.linalg.diag(eig.eigenvalues) @ eig.inverse_eigenvectors
    )
    q_norm = model.q_norm(frequencies=pi, rates=rates)
    assert_allclose(q_reconstructed.numpy(), q_norm.numpy(), atol=1e-10)


def test_batched_rates(tensor_constant):
    """Support a leading batch dimension on rates and frequencies."""
    K = 4
    batch = 3
    model = DiscreteTraitModel(K)
    rng = np.random.default_rng(0)
    pi = rng.dirichlet(np.ones(K), size=batch)
    rates = rng.uniform(0.1, 2.0, size=(batch, K * (K - 1) // 2))

    q = model.q(
        frequencies=tensor_constant(pi), rates=tensor_constant(rates)
    ).numpy()

    assert q.shape == (batch, K, K)
    for b in range(batch):
        row_sums = q[b].sum(axis=-1)
        assert_allclose(row_sums, np.zeros(K), atol=1e-10)
        # reversibility per batch element
        lhs = pi[b][:, None] * q[b]
        assert_allclose(lhs, lhs.T, atol=1e-10)


def test_invalid_n_states():
    with pytest.raises(ValueError):
        DiscreteTraitModel(1)
