import pytest
from treeflow_test_helpers.substitution_helpers import (
    EigenSubstitutionModelHelper,
)
from treeflow.evolution.substitution.nucleotide.gtr import GTR
from treeflow.evolution.substitution.nucleotide.hky import HKY
import numpy as np
from numpy.testing import assert_allclose


@pytest.fixture
def gtr_params(tensor_constant):
    return dict(
        frequencies=tensor_constant([0.21, 0.28, 0.27, 0.24]),
        rates=tensor_constant([0.2, 0.12, 0.17, 0.09, 0.24, 0.18]),
    )


class TestGTR(EigenSubstitutionModelHelper):
    ClassUnderTest = GTR

    def _init(self, gtr_params):
        self.params = gtr_params

    def test_eigendecomposition(self, gtr_params):
        self._init(gtr_params)
        super().test_eigendecomposition()

    def test_hky_special_case(self, tensor_constant, hky_params):
        kappa = hky_params["kappa"].numpy()
        rates = np.ones(6)
        rates[1] = kappa
        rates[4] = kappa
        rates = tensor_constant(rates / np.sum(rates))
        gtr_q_norm = GTR().q_norm(frequencies=hky_params["frequencies"], rates=rates)
        hky_q_norm = HKY().q_norm(**hky_params)
        assert_allclose(gtr_q_norm.numpy(), hky_q_norm.numpy())
