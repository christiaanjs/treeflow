"""Tests for the clade-model straight-through phylogenetic likelihood."""

import numpy as np
import pytest
import tensorflow as tf

from treeflow import DEFAULT_FLOAT_DTYPE_TF as FLOAT
from treeflow.conditional_clade.clade_likelihood import (
    clade_straight_through_log_likelihood,
    exhaustive_candidates,
    sampled_candidates,
    sampled_subtree_partial_fn,
)
from treeflow.conditional_clade.distribution import ConditionalCladeDistribution
from treeflow.conditional_clade.support import ConditionalCladeSupport
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology


def _setup(n=5, state=4, sites=6, seed=0):
    rng = np.random.default_rng(seed)
    support = ConditionalCladeSupport(n)
    logits = tf.Variable(tf.constant(rng.standard_normal(support.subsplit_count), FLOAT))
    q = ConditionalCladeDistribution(support, logits)
    parent_indices = q.sample_parent_indices(rng)
    transition = tf.constant(rng.dirichlet(np.ones(state), size=state), FLOAT)
    frequencies = tf.constant(np.ones(state) / state, FLOAT)
    sequences = tf.constant(np.eye(state)[rng.integers(0, state, size=(sites, n))], FLOAT)
    return dict(
        n=n, state=state, support=support, logits=logits, q=q,
        parent_indices=parent_indices, transition=transition,
        frequencies=frequencies, sequences=sequences,
    )


@pytest.mark.parametrize("n", [4, 5])
def test_forward_matches_felsenstein(n):
    s = _setup(n=n, seed=n)
    Pn, freq, seq = (
        s["transition"].numpy(), s["frequencies"].numpy(), s["sequences"].numpy()
    )
    child = NumpyTreeTopology(parent_indices=s["parent_indices"]).child_indices
    partial = {i: seq[:, i, :] for i in range(n)}
    for u in range(n, 2 * n - 1):
        c0, c1 = child[u]
        partial[u] = (partial[c0] @ Pn.T) * (partial[c1] @ Pn.T)
    manual = np.log(partial[2 * n - 2] @ freq)
    forward = clade_straight_through_log_likelihood(
        s["q"], s["parent_indices"], s["sequences"], s["transition"], s["frequencies"]
    ).numpy()
    np.testing.assert_allclose(forward, manual)


def test_gradient_reaches_clade_logits():
    s = _setup(seed=1)
    with tf.GradientTape() as tape:
        log_lik = tf.reduce_sum(
            clade_straight_through_log_likelihood(
                s["q"], s["parent_indices"], s["sequences"],
                s["transition"], s["frequencies"],
            )
        )
    grad = tape.gradient(log_lik, s["logits"]).numpy()
    assert np.all(np.isfinite(grad))
    assert np.any(grad != 0)


def test_gather_routing_matches_dense():
    s = _setup(seed=2)
    leaf_partials = {1 << i: s["sequences"][..., i, :] for i in range(s["n"])}
    alt = sampled_subtree_partial_fn(s["q"], leaf_partials, s["transition"], seed=7)

    def run(gather):
        with tf.GradientTape() as tape:
            log_lik = tf.reduce_sum(
                clade_straight_through_log_likelihood(
                    s["q"], s["parent_indices"], s["sequences"],
                    s["transition"], s["frequencies"],
                    candidate_subsplits_fn=exhaustive_candidates,
                    alternative_partial_fn=alt, gather=gather,
                )
            )
        return log_lik.numpy(), tape.gradient(log_lik, s["logits"]).numpy()

    value_gather, grad_gather = run(True)
    value_dense, grad_dense = run(False)
    np.testing.assert_allclose(value_gather, value_dense)
    np.testing.assert_allclose(grad_gather, grad_dense)


def test_runs_for_sampled_candidate_set():
    s = _setup(seed=3)
    with tf.GradientTape() as tape:
        log_lik = tf.reduce_sum(
            clade_straight_through_log_likelihood(
                s["q"], s["parent_indices"], s["sequences"],
                s["transition"], s["frequencies"],
                candidate_subsplits_fn=sampled_candidates(2, seed=1),
            )
        )
    grad = tape.gradient(log_lik, s["logits"]).numpy()
    assert np.all(np.isfinite(grad))
    assert np.any(grad != 0)


def test_sampled_candidates_forward_is_exact():
    """A sampled candidate set must not change the forward likelihood (the
    realised subsplit is always selected)."""
    s = _setup(seed=4)
    full = clade_straight_through_log_likelihood(
        s["q"], s["parent_indices"], s["sequences"], s["transition"],
        s["frequencies"], candidate_subsplits_fn=exhaustive_candidates,
    ).numpy()
    sampled = clade_straight_through_log_likelihood(
        s["q"], s["parent_indices"], s["sequences"], s["transition"],
        s["frequencies"], candidate_subsplits_fn=sampled_candidates(1, seed=5),
    ).numpy()
    np.testing.assert_allclose(full, sampled)
