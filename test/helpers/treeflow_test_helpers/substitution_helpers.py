from __future__ import annotations
import typing as tp
import tensorflow as tf
from treeflow.evolution.substitution.base_substitution_model import (
    EigendecompositionSubstitutionModel,
)
from numpy.testing import assert_allclose


class SubstitutionModelHelper:

    ClassUnderTest: tp.Type[EigendecompositionSubstitutionModel]
    params: tp.Mapping[str, tf.Tensor]

    # TODO


class EigenSubstitutionModelHelper(SubstitutionModelHelper):
    def test_eigendecomposition(self):
        model = self.ClassUnderTest()
        res = model.eigen(**self.params)
        assert res.eigenvalues.shape == (4,)
        assert res.eigenvectors.shape == (4, 4)
        assert res.inverse_eigenvectors.shape == (4, 4)

        q = model.q_norm(**self.params)
        q_res = (
            res.eigenvectors
            @ tf.linalg.diag(res.eigenvalues)
            @ res.inverse_eigenvectors
        )
        assert_allclose(q.numpy(), q_res.numpy())
