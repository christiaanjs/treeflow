"""Shared fixtures for VBPI subsplit-Bayesian-network tests."""
import numpy as np
import pytest


def random_parent_indices(taxon_count: int, rng: np.random.Generator) -> np.ndarray:
    """A uniformly-drawn random rooted binary topology (treeflow convention).

    Leaves are ``0..n-1``; internal nodes are created in coalescent order and so
    are numbered increasingly, giving the children<parent, root-last layout.
    """
    node_count = 2 * taxon_count - 1
    parent = np.full(node_count - 1, -1, dtype=np.int32)
    active = list(range(taxon_count))
    next_node = taxon_count
    while len(active) > 1:
        i = active.pop(rng.integers(len(active)))
        j = active.pop(rng.integers(len(active)))
        parent[i] = next_node
        parent[j] = next_node
        active.append(next_node)
        next_node += 1
    # ``parent`` has length 2n-2 (no root entry); the last coalescence created
    # the root (id 2n-2), which correctly has no parent slot.
    return parent


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
