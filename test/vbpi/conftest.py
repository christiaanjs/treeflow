"""Shared fixtures for VBPI subsplit-Bayesian-network tests."""
import numpy as np
import pytest

from vbpi_test_helpers import random_parent_indices


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def random_topologies(rng):
    """A modest collection of random 7-taxon topologies (with repeats likely)."""
    return [random_parent_indices(7, rng) for _ in range(40)]


@pytest.fixture
def make_random_topologies():
    def _make(taxon_count, count, seed=0):
        r = np.random.default_rng(seed)
        return [random_parent_indices(taxon_count, r) for _ in range(count)]

    return _make
