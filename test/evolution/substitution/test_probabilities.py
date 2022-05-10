from cgitb import reset
import pytest
from numpy.testing import assert_allclose
import tensorflow as tf
from treeflow.evolution.substitution.nucleotide.hky import HKY
from treeflow.evolution.substitution.nucleotide.jc import JC
from treeflow.evolution.substitution.probabilities import (
    get_transition_probabilities_eigen,
    get_transition_probabilities_tree,
)


_branch_lengths = [0.1, [1.2, 3.2, 0.1], [[0.05, 1.3], [0.8, 0.3]]]


@pytest.fixture(params=_branch_lengths)
def branch_lengths(tensor_constant, request):
    return tensor_constant(request.param)


def test_get_transition_probabilities_eigen_hky_rowsum(
    branch_lengths,
    hky_params,
):
    eigen_batch = (
        HKY().eigen(**hky_params).add_inner_batch_dimensions(branch_lengths.shape.rank)
    )

    res = get_transition_probabilities_eigen(eigen_batch, branch_lengths)
    row_sums = tf.reduce_sum(res, axis=-1)
    assert_allclose(1.0, row_sums)


def test_get_transition_probabilities_tree_hky_vec(
    hky_params_vec, hello_tensor_tree, tensor_constant
):
    rate_categories = tensor_constant([0.1, 0.3, 0.6, 1.0, 1.5])
    category_count = rate_categories.shape[0]
    unrooted_tree = hello_tensor_tree.get_unrooted_tree()
    res = get_transition_probabilities_tree(
        unrooted_tree,
        HKY(),
        **hky_params_vec,
        rate_categories=rate_categories,
        batch_rank=-1,
        inner_batch_rank=1
    )
    assert tuple(res.branch_lengths.shape) == (category_count,) + tuple(
        hky_params_vec["kappa"].shape
    ) + (
        4,
        4,
    )
