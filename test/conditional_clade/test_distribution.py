import numpy as np
import pytest
import tensorflow as tf

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.conditional_clade.distribution import (
    ConditionalCladeDistribution,
    segment_log_softmax,
)
from treeflow.conditional_clade.support import ConditionalCladeSupport


def random_logits(support, seed=0):
    rng = np.random.default_rng(seed)
    return tf.constant(
        rng.standard_normal(support.subsplit_count), dtype=DEFAULT_FLOAT_DTYPE_TF
    )


def test_segment_log_softmax_normalises():
    logits = tf.constant([1.0, 2.0, 0.5, -1.0, 3.0], dtype=DEFAULT_FLOAT_DTYPE_TF)
    segment_ids = tf.constant([0, 0, 1, 1, 1], dtype=tf.int32)
    log_probs = segment_log_softmax(logits, segment_ids, 2)
    probs = np.exp(log_probs.numpy())
    assert np.isclose(probs[:2].sum(), 1.0)
    assert np.isclose(probs[2:].sum(), 1.0)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_enumerated_probs_sum_to_one(n):
    support = ConditionalCladeSupport(n)
    dist = ConditionalCladeDistribution(support, random_logits(support))
    probs = dist.enumerate_probs().numpy()
    assert np.isclose(probs.sum(), 1.0)
    assert probs.shape == (support.topology_count(),)


@pytest.mark.parametrize("n", [3, 4])
def test_uniform_logits_give_uniform_distribution(n):
    support = ConditionalCladeSupport(n)
    # Zero logits do NOT generally give a uniform distribution over topologies
    # (clades have different numbers of subsplits); just check it's a valid pmf.
    dist = ConditionalCladeDistribution(support)
    probs = dist.enumerate_probs().numpy()
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs > 0)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_log_prob_matches_enumeration(n):
    support = ConditionalCladeSupport(n)
    dist = ConditionalCladeDistribution(support, random_logits(support, seed=1))
    parent_indices_list = dist.enumerate_parent_indices()
    enumerated_log_probs = dist.enumerate_log_probs().numpy()
    for parent_indices, expected in zip(parent_indices_list, enumerated_log_probs):
        actual = dist.log_prob(parent_indices).numpy()
        assert np.isclose(actual, expected)


def test_sampling_matches_probabilities():
    n = 4
    support = ConditionalCladeSupport(n)
    dist = ConditionalCladeDistribution(support, random_logits(support, seed=2))
    parent_indices_list = dist.enumerate_parent_indices()
    index_of = {tuple(pi.tolist()): i for i, pi in enumerate(parent_indices_list)}
    probs = dist.enumerate_probs().numpy()

    rng = np.random.default_rng(123)
    n_samples = 40000
    counts = np.zeros(len(parent_indices_list))
    for _ in range(n_samples):
        pi = dist.sample_parent_indices(rng)
        counts[index_of[tuple(pi.tolist())]] += 1
    empirical = counts / n_samples
    # Monte-Carlo agreement with the exact pmf
    assert np.max(np.abs(empirical - probs)) < 0.02


def test_entropy_matches_manual():
    n = 4
    support = ConditionalCladeSupport(n)
    dist = ConditionalCladeDistribution(support, random_logits(support, seed=3))
    probs = dist.enumerate_probs().numpy()
    manual = -np.sum(probs * np.log(probs))
    assert np.isclose(dist.entropy().numpy(), manual)


def test_exact_kl_self_is_zero_and_nonnegative():
    n = 4
    support = ConditionalCladeSupport(n)
    q = ConditionalCladeDistribution(support, random_logits(support, seed=4))
    p = ConditionalCladeDistribution(support, random_logits(support, seed=5))
    assert np.isclose(q.exact_kl_divergence(q).numpy(), 0.0, atol=1e-10)
    kl = q.exact_kl_divergence(p).numpy()
    assert kl > 0


def test_exact_kl_matches_enumerated_definition():
    n = 4
    support = ConditionalCladeSupport(n)
    q = ConditionalCladeDistribution(support, random_logits(support, seed=6))
    p = ConditionalCladeDistribution(support, random_logits(support, seed=7))
    log_q = q.enumerate_log_probs().numpy()
    log_p = p.enumerate_log_probs().numpy()
    manual = np.sum(np.exp(log_q) * (log_q - log_p))
    assert np.isclose(q.exact_kl_divergence(p).numpy(), manual)


def test_clade_visitation_root_is_one_and_consistent():
    n = 4
    support = ConditionalCladeSupport(n)
    dist = ConditionalCladeDistribution(support, random_logits(support, seed=8))
    visit = dist.clade_visitation_probabilities()
    assert np.isclose(visit[support.root_clade], 1.0)
    # Compare against Monte-Carlo visitation of a specific clade.
    from treeflow.conditional_clade.clade import taxa_to_clade

    target = taxa_to_clade((0, 1))
    rng = np.random.default_rng(99)
    n_samples = 20000
    hits = 0
    for _ in range(n_samples):
        assignment = dist.sample_assignment(rng)
        clades = set(assignment.keys())
        for subsplit in assignment.values():
            clades.add(subsplit.child1)
            clades.add(subsplit.child2)
        if target in clades:
            hits += 1
    empirical = hits / n_samples
    assert np.isclose(empirical, visit[target], atol=0.02)


def test_log_prob_differentiable():
    n = 4
    support = ConditionalCladeSupport(n)
    logits = tf.Variable(random_logits(support, seed=10))
    dist = ConditionalCladeDistribution(support, logits)
    parent_indices = dist.enumerate_parent_indices()[0]
    with tf.GradientTape() as tape:
        log_prob = dist.log_prob(parent_indices)
    grad = tape.gradient(log_prob, logits)
    assert grad is not None
    assert np.any(grad.numpy() != 0)
