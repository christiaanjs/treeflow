"""Correctness tests for the native SBN topology-sampler custom op.

The native sampler is checked against the pure-NumPy reference walk in
:class:`treeflow.vbpi.sbn.SubsplitBayesianNetwork`: every sampled topology must
be valid and in support, its recorded candidate indices must match those
recomputed from the topology, and (over many draws) the empirical topology
frequencies must match the analytic ``log_prob`` -- the same statistical anchor
the reference sampler is held to.
"""
from collections import Counter

import numpy as np
import pytest
import tensorflow as tf

from treeflow.vbpi.sbn import SubsplitBayesianNetwork
from treeflow.vbpi.support import SubsplitSupport, all_rooted_parent_indices


def _sbn(taxon_count, seed=0):
    topos = all_rooted_parent_indices(taxon_count)
    support = SubsplitSupport.from_topologies(topos, taxon_count)
    sbn = SubsplitBayesianNetwork(support)
    sbn.logits.assign(
        tf.constant(np.random.default_rng(seed).normal(size=support.num_candidates))
    )
    return sbn, topos


def _empirical(sbn, samples, topos):
    cand_all = sbn.support.batch_topology_candidate_indices(topos)
    key_to_index = {tuple(sorted(ci)): t for t, ci in enumerate(cand_all)}
    counts = np.zeros(len(topos))
    for row in samples.candidate_indices:
        counts[key_to_index[tuple(sorted(row))]] += 1
    return counts / counts.sum()


def test_native_available():
    from treeflow.acceleration.native import sbn as native_sbn

    assert native_sbn.is_available()


@pytest.mark.parametrize("taxon_count", [4, 5, 6])
def test_native_samples_are_valid_topologies(taxon_count):
    sbn, _ = _sbn(taxon_count, seed=taxon_count)
    samples = sbn.sample_topologies(200, seed=1, use_native=True)
    n = taxon_count
    for i in range(200):
        pi = samples.parent_indices[i]
        # children < parent, root id 2n-2
        for node, parent in enumerate(pi):
            assert parent > node
        assert pi.max() == 2 * n - 2
        # candidate indices recomputed from the topology match the recorded ones
        recomputed = sbn.support.topology_candidate_indices(pi)
        assert set(recomputed) == set(samples.candidate_indices[i])


def test_native_node_clade_ids_consistent():
    sbn, _ = _sbn(6, seed=2)
    samples = sbn.sample_topologies(50, seed=3, use_native=True)
    n = sbn.support.taxon_count
    for i in range(50):
        for taxon in range(n):
            assert sbn.support.clade_leaf_taxon[samples.node_clade_ids[i, taxon]] == taxon
        assert samples.node_clade_ids[i, 2 * n - 2] == sbn.support.root_clade_id


def test_native_frequencies_match_log_prob():
    sbn, topos = _sbn(5, seed=4)
    samples = sbn.sample_topologies(80000, seed=9, use_native=True)
    freqs = _empirical(sbn, samples, topos)
    probs = tf.exp(sbn.log_prob_of_topologies(topos)).numpy()
    assert np.max(np.abs(freqs - probs)) < 0.01


def test_native_matches_numpy_distribution():
    """Native and NumPy samplers realise the *same* distribution (not the same
    draws): their empirical frequencies agree to Monte-Carlo error."""
    sbn, topos = _sbn(5, seed=5)
    native = sbn.sample_topologies(80000, seed=1, use_native=True)
    numpy = sbn.sample_topologies(80000, seed=1, use_native=False)
    f_native = _empirical(sbn, native, topos)
    f_numpy = _empirical(sbn, numpy, topos)
    assert np.max(np.abs(f_native - f_numpy)) < 0.015


def test_native_seed_reproducible():
    sbn, _ = _sbn(5, seed=6)
    a = sbn.sample_topologies(100, seed=42, use_native=True)
    b = sbn.sample_topologies(100, seed=42, use_native=True)
    assert np.array_equal(a.parent_indices, b.parent_indices)
    assert np.array_equal(a.candidate_indices, b.candidate_indices)


def test_native_auto_resolves_to_native():
    sbn, _ = _sbn(5, seed=7)
    # 'auto' should pick the native path since the op is built in this env.
    assert sbn._resolve_native("auto") is True
