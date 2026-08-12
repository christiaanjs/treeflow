from collections import Counter

import numpy as np
import pytest
import tensorflow as tf

from treeflow.vbpi.sbn import SubsplitBayesianNetwork
from treeflow.vbpi.support import SubsplitSupport, all_rooted_parent_indices


@pytest.fixture
def complete_sbn():
    """SBN with complete 5-taxon support and random logits."""
    n = 5
    topos = all_rooted_parent_indices(n)
    support = SubsplitSupport.from_topologies(topos, n)
    sbn = SubsplitBayesianNetwork(support)
    sbn.logits.assign(
        tf.constant(np.random.default_rng(0).normal(size=support.num_candidates))
    )
    return sbn, topos


def test_conditional_probs_normalise(complete_sbn):
    sbn, _ = complete_sbn
    probs = sbn.conditional_probs().numpy()
    # sum within each parent group is one
    for c in range(sbn.support.num_clades):
        start = sbn.support.child_offsets[c]
        end = sbn.support.child_offsets[c + 1]
        if end > start:
            assert np.isclose(probs[start:end].sum(), 1.0)


@pytest.mark.parametrize("taxon_count", [4, 5, 6])
def test_log_prob_normalises_over_all_topologies(taxon_count):
    topos = all_rooted_parent_indices(taxon_count)
    support = SubsplitSupport.from_topologies(topos, taxon_count)
    sbn = SubsplitBayesianNetwork(support)
    sbn.logits.assign(
        tf.constant(
            np.random.default_rng(taxon_count).normal(size=support.num_candidates)
        )
    )
    lp = sbn.log_prob_of_topologies(topos)
    total = tf.reduce_sum(tf.exp(lp)).numpy()
    assert np.isclose(total, 1.0, atol=1e-9)


def test_log_prob_gradient_flows(complete_sbn):
    sbn, topos = complete_sbn
    cand = tf.constant(sbn.support.batch_topology_candidate_indices(topos))
    with tf.GradientTape() as tape:
        lp = sbn.log_prob(cand)
        loss = tf.reduce_sum(lp)
    grad = tape.gradient(loss, sbn.logits)
    assert grad is not None
    assert grad.shape == sbn.logits.shape


def test_from_topologies_initialises_ccd_estimate():
    # A biased collection: with pseudo_count small, the empirical CCD should put
    # most mass on the observed topology.
    n = 5
    topos = all_rooted_parent_indices(n)
    biased = [topos[0]] * 50 + [topos[1]]
    sbn = SubsplitBayesianNetwork.from_topologies(
        biased, n, pseudo_count=1e-3
    )
    lp = sbn.log_prob_of_topologies([topos[0], topos[1]]).numpy()
    assert lp[0] > lp[1]
    assert np.exp(lp[0]) > 0.5


def _empirical_frequencies(sbn, samples, topos):
    cand_all = sbn.support.batch_topology_candidate_indices(topos)
    key_to_index = {tuple(sorted(ci)): t for t, ci in enumerate(cand_all)}
    counts = np.zeros(len(topos))
    for row in samples.candidate_indices:
        counts[key_to_index[tuple(sorted(row))]] += 1
    return counts / counts.sum()


def test_numpy_sampler_matches_log_prob(complete_sbn):
    sbn, topos = complete_sbn
    samples = sbn.sample_topologies(80000, seed=11, use_native=False)
    freqs = _empirical_frequencies(sbn, samples, topos)
    probs = tf.exp(sbn.log_prob_of_topologies(topos)).numpy()
    assert np.max(np.abs(freqs - probs)) < 0.01


def test_sampled_candidate_indices_recompute_from_topology(complete_sbn):
    sbn, _ = complete_sbn
    samples = sbn.sample_topologies(25, seed=5, use_native=False)
    for i in range(25):
        recomputed = sbn.support.topology_candidate_indices(
            samples.parent_indices[i]
        )
        assert set(recomputed) == set(samples.candidate_indices[i])


def test_sampled_node_clade_ids_consistent(complete_sbn):
    sbn, _ = complete_sbn
    samples = sbn.sample_topologies(10, seed=6, use_native=False)
    n = sbn.support.taxon_count
    for i in range(10):
        # leaves: node id == taxon, clade is the singleton for that taxon
        for taxon in range(n):
            clade_id = samples.node_clade_ids[i, taxon]
            assert sbn.support.clade_leaf_taxon[clade_id] == taxon
        # root node clade is the root clade
        root_node = 2 * n - 2
        assert samples.node_clade_ids[i, root_node] == sbn.support.root_clade_id
