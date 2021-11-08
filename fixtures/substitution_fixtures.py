import pytest


@pytest.fixture
def hky_params(tensor_constant):
    return dict(
        frequencies=tensor_constant([0.23, 0.27, 0.24, 0.26]),
        kappa=tensor_constant(2.0),
    )
