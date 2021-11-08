import tensorflow as tf
from treeflow.evolution.substitution.eigendecomposition import Eigendecomposition


def test_eigendecomposition_add_inner_batch_dims():
    n_states = 5
    batch_shape = (3, 2)
    eig = Eigendecomposition(
        eigenvectors=tf.zeros(batch_shape + (n_states, n_states)),
        inverse_eigenvectors=tf.zeros(batch_shape + (n_states, n_states)),
        eigenvalues=tf.zeros(batch_shape + (n_states,)),
    )
    res = eig.add_inner_batch_dimensions(2)
    assert res.eigenvectors.numpy().shape == batch_shape + (1, 1) + (n_states, n_states)
    assert res.inverse_eigenvectors.numpy().shape == batch_shape + (1, 1) + (
        n_states,
        n_states,
    )
    assert res.eigenvalues.numpy().shape == batch_shape + (1, 1) + (n_states,)
