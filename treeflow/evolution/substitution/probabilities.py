import tensorflow as tf
from treeflow.evolution.substitution.eigendecomposition import Eigendecomposition
from treeflow.evolution.substitution.base_substitution_model import (
    SubstitutionModel,
    EigendecompositionSubstitutionModel,
)
from treeflow.tree.unrooted.tensorflow_unrooted_tree import TensorflowUnrootedTree


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


def get_batch_transition_probabilities_eigen(
    eigen: Eigendecomposition, t: tf.Tensor, batch_rank: int = 1
) -> tf.Tensor:
    """
    Get transition probabilities for an eigendecomposition of a rate matrix and
    a time (i.e. genetic distances in expected substitutions per site).

    Parameters
    ----------
    eigen
        Eigendecomposition of Tensors (can have batch dimensions ...)
    t
        Tensor of times with shape (..., branches)

    Returns
    -------
    Tensor
        Transition probabilities with shape (..., branches)
    """
    eigen_with_batch = eigen.add_inner_batch_dimensions(batch_rank)
    return get_transition_probabilities_eigen(eigen_with_batch, t)


def get_transition_probabilities_tree(
    tree: TensorflowUnrootedTree,
    subst_model: SubstitutionModel,
    batch_rank: int = 0,
    **subst_params
) -> TensorflowUnrootedTree:
    if isinstance(subst_model, EigendecompositionSubstitutionModel):
        return tree.with_branch_lengths(
            get_batch_transition_probabilities_eigen(
                subst_model.eigen(**subst_params),
                tree.branch_lengths,
                batch_rank=batch_rank + 1,
            )
        )
    else:
        raise ValueError("Only implemented for eigen substitution model")


__all__ = [
    get_transition_probabilities_eigen.__name__,
    get_batch_transition_probabilities_eigen.__name__,
    get_transition_probabilities_tree.__name__,
]
