"""Shared helpers for VBPI tests.

Kept in a uniquely-named module (not ``conftest``) so the test files can import
it directly without colliding with other packages' ``conftest.py`` when the
suites are collected together.
"""
import numpy as np


def random_parent_indices(taxon_count, rng):
    """A uniformly-drawn random rooted binary topology (treeflow convention)."""
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
    return parent


def simulate_jc_alignment(parent_indices, branch_lengths, n_sites, taxon_count, rng):
    """Simulate a JC nucleotide alignment down a rooted tree.

    Returns leaf partials ``[n, n_sites, 4]`` (one-hot). Root states are uniform;
    each branch mutates under the JC transition matrix.
    """
    node_count = 2 * taxon_count - 1

    def P(t):
        r = np.exp(-4.0 / 3.0 * t)
        return np.where(np.eye(4, dtype=bool), 0.25 + 0.75 * r, 0.25 - 0.25 * r)

    states = np.full((node_count, n_sites), -1, dtype=int)
    states[node_count - 1] = rng.integers(0, 4, size=n_sites)  # root
    for node in range(node_count - 2, -1, -1):  # parents before children
        probs = P(branch_lengths[node])[states[parent_indices[node]]]
        u = rng.random((n_sites, 1))
        states[node] = (np.cumsum(probs, axis=1) > u).argmax(axis=1)
    return np.eye(4)[states[:taxon_count]]
