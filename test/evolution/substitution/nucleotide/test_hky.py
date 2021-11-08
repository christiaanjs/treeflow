import pytest
from treeflow.evolution.substitution.nucleotide.hky import (
    HKY,
    pack_matrix,
    pack_matrix_transposed,
)
from functools import reduce
import tensorflow as tf
from numpy.testing import assert_allclose


@pytest.mark.parametrize("batch_shape", [(1, 2), ()])
def test_pack_matrix(batch_shape):
    nrow = 3
    ncol = 2
    batch_size = reduce(lambda x, y: x * y, batch_shape, 1)
    elements = [
        [
            tf.reshape(tf.range(i + j, i + j + batch_size), batch_shape)
            for j in range(ncol)
        ]
        for i in range(nrow)
    ]
    res = pack_matrix(elements)
    for i in range(nrow):
        for j in range(ncol):
            element = res[..., i, j]
            assert element.shape == batch_shape
            assert_allclose(elements[i][j], element.numpy())


@pytest.mark.parametrize("batch_shape", [(1, 2), ()])
def test_pack_matrix_transpose(batch_shape):
    nrow = 3
    ncol = 2
    batch_size = reduce(lambda x, y: x * y, batch_shape, 1)
    elements = [
        [
            tf.reshape(tf.range(i + j, i + j + batch_size), batch_shape)
            for j in range(ncol)
        ]
        for i in range(nrow)
    ]
    res = pack_matrix_transposed(elements)
    for i in range(nrow):
        for j in range(ncol):
            element = res[..., j, i]
            assert element.shape == batch_shape
            assert_allclose(elements[i][j], element.numpy())


def test_hky_eigendecomposition(hky_params):
    res = HKY().eigen(**hky_params)
    assert res.eigenvalues.shape == (4,)
    assert res.eigenvectors.shape == (4, 4)
    assert res.inverse_eigenvectors.shape == (4, 4)


# TODO: Test for batch
