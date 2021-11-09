import tensorflow as tf
from treeflow.evolution.substitution.eigendecomposition import Eigendecomposition


def get_transition_probabilities_eigen(
    eigen: Eigendecomposition, t: tf.Tensor
) -> tf.Tensor:
    """
    Get transition probabilities for an eigendecomposition of a rate matrix and
    a time (i.e. genetic distances in expected substitutions per site).

    Parameters
    ----------
    eigen
        Eigendecomposition of Tensors (can have batch dimensions)
    t
        Time Tensor (can be batched)

    Returns
    -------
    Tensor
        Transition probabilities with shape [..., state, state] where ... is batch shape
    """
    diag = tf.linalg.diag(tf.exp(tf.expand_dims(t, -1) * eigen.eigenvalues))
    return eigen.eigenvectors @ diag @ eigen.inverse_eigenvectors
