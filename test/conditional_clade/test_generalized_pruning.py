"""Tests for generalized pruning over the subsplit DAG.

The defining property: the DAG sum-product equals the per-site tree-marginal
``sum_s log E_{q(T)}[L_s(T)]`` computed by brute force over every enumerated tree
-- in both value and gradient.
"""

import numpy as np
import pytest
import tensorflow as tf

from treeflow import DEFAULT_FLOAT_DTYPE_TF as FLOAT
from treeflow.conditional_clade.clade import is_singleton
from treeflow.conditional_clade.distribution import ConditionalCladeDistribution
from treeflow.conditional_clade.generalized_pruning import (
    SubsplitDAG,
    build_tip_partials,
    build_transition_matrices,
    exact_weights,
    gumbel_softmax_weights,
    relaxed_log_likelihood,
    relaxed_log_likelihood_from_distribution,
    relaxed_partials_sequential,
    relaxed_partials_vectorized,
    straight_through_weights,
)
from treeflow.conditional_clade.support import ConditionalCladeSupport
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology


def _setup(n=5, state=4, sites=6, seed=0):
    rng = np.random.default_rng(seed)
    support = ConditionalCladeSupport(n)
    logits = tf.Variable(tf.constant(rng.standard_normal(support.subsplit_count), FLOAT))
    q = ConditionalCladeDistribution(support, logits)
    dag = SubsplitDAG(support)
    P = tf.constant(rng.dirichlet(np.ones(state), size=state), FLOAT)  # [state, state]
    P_clades = tf.constant(np.tile(P.numpy(), (dag.num_clades, 1, 1)), FLOAT)
    frequencies = tf.constant(np.ones(state) / state, FLOAT)
    sequences = tf.constant(np.eye(state)[rng.integers(0, state, size=(sites, n))], FLOAT)
    return dict(n=n, state=state, support=support, logits=logits, q=q, dag=dag,
                P=P, P_clades=P_clades, frequencies=frequencies, sequences=sequences)


def _brute_force_marginal(s):
    """sum_s log( sum_T q(T) L_s(T) ) over enumerated trees (differentiable in q)."""
    Pn, freq, seq = s["P"].numpy(), s["frequencies"].numpy(), s["sequences"].numpy()
    n = s["n"]
    trees = s["q"].enumerate_parent_indices()
    site_liks = []
    for pi in trees:
        child = NumpyTreeTopology(parent_indices=pi).child_indices
        part = {i: seq[:, i, :] for i in range(n)}
        for u in range(n, 2 * n - 1):
            c0, c1 = int(child[u][0]), int(child[u][1])
            part[u] = (part[c0] @ Pn.T) * (part[c1] @ Pn.T)
        site_liks.append(part[2 * n - 2] @ freq)  # [sites]
    L = tf.constant(np.stack(site_liks), FLOAT)  # [trees, sites]
    probs = s["q"].enumerate_probs()  # [trees], differentiable
    per_site = tf.reduce_sum(probs[:, tf.newaxis] * L, axis=0)  # [sites]
    return tf.reduce_sum(tf.math.log(per_site))


@pytest.mark.parametrize("n", [3, 4, 5])
def test_sequential_matches_brute_force(n):
    s = _setup(n=n, seed=n)
    tip = build_tip_partials(s["dag"], s["sequences"])
    w = tf.exp(s["q"].conditional_log_probs())
    relaxed = relaxed_log_likelihood(
        s["dag"], w, s["P_clades"], tip, s["frequencies"]
    ).numpy()
    brute = _brute_force_marginal(s).numpy()
    np.testing.assert_allclose(relaxed, brute, rtol=1e-10)


def test_gradient_matches_brute_force():
    s = _setup(seed=1)
    tip = build_tip_partials(s["dag"], s["sequences"])
    with tf.GradientTape() as tape:
        w = tf.exp(s["q"].conditional_log_probs())
        relaxed = relaxed_log_likelihood(s["dag"], w, s["P_clades"], tip, s["frequencies"])
    grad_relaxed = tape.gradient(relaxed, s["logits"]).numpy()

    with tf.GradientTape() as tape:
        brute = _brute_force_marginal(s)
    grad_brute = tape.gradient(brute, s["logits"]).numpy()

    assert np.all(np.isfinite(grad_relaxed))
    assert np.any(grad_relaxed != 0)
    np.testing.assert_allclose(grad_relaxed, grad_brute, rtol=1e-8, atol=1e-10)


def test_vectorized_matches_sequential():
    s = _setup(seed=2)
    tip = build_tip_partials(s["dag"], s["sequences"])
    w = tf.exp(s["q"].conditional_log_probs())
    seq_partials = relaxed_partials_sequential(s["dag"], w, s["P_clades"], tip).numpy()
    vec_partials = relaxed_partials_vectorized(s["dag"], w, s["P_clades"], tip).numpy()
    np.testing.assert_allclose(seq_partials, vec_partials, rtol=1e-10)


def test_vectorized_gradient_matches_sequential():
    s = _setup(seed=3)
    tip = build_tip_partials(s["dag"], s["sequences"])

    def run(vectorized):
        with tf.GradientTape() as tape:
            w = tf.exp(s["q"].conditional_log_probs())
            ll = relaxed_log_likelihood(
                s["dag"], w, s["P_clades"], tip, s["frequencies"], vectorized=vectorized
            )
        return ll.numpy(), tape.gradient(ll, s["logits"]).numpy()

    v_seq, g_seq = run(False)
    v_vec, g_vec = run(True)
    np.testing.assert_allclose(v_seq, v_vec, rtol=1e-10)
    np.testing.assert_allclose(g_seq, g_vec, rtol=1e-8, atol=1e-12)


def test_runs_in_graph_mode():
    s = _setup(seed=4)
    tip = build_tip_partials(s["dag"], s["sequences"])

    @tf.function
    def graph_ll():
        w = tf.exp(s["q"].conditional_log_probs())
        return relaxed_log_likelihood(s["dag"], w, s["P_clades"], tip, s["frequencies"])

    eager = relaxed_log_likelihood(
        s["dag"], tf.exp(s["q"].conditional_log_probs()), s["P_clades"], tip, s["frequencies"]
    ).numpy()
    np.testing.assert_allclose(graph_ll().numpy(), eager, rtol=1e-10)


def test_per_clade_transitions_consistent():
    """Vectorized and sequential agree with genuinely per-clade (varying) P."""
    s = _setup(seed=5)
    rng = np.random.default_rng(99)
    dag = s["dag"]
    mats = rng.dirichlet(np.ones(s["state"]), size=(dag.num_clades, s["state"]))
    P_clades = tf.constant(mats, FLOAT)  # [num_clades, state, state]
    tip = build_tip_partials(dag, s["sequences"])
    w = tf.exp(s["q"].conditional_log_probs())
    seq = relaxed_partials_sequential(dag, w, P_clades, tip).numpy()
    vec = relaxed_partials_vectorized(dag, w, P_clades, tip).numpy()
    np.testing.assert_allclose(seq, vec, rtol=1e-9)


def test_build_transition_matrices_from_fn():
    s = _setup(seed=6)
    P = s["P"]
    P_clades = build_transition_matrices(s["dag"], lambda clade: P)
    # every non-root row equals P; root is identity
    for row in range(s["dag"].num_clades):
        expected = np.eye(s["state"]) if row == s["dag"].root_index else P.numpy()
        np.testing.assert_allclose(P_clades[row].numpy(), expected)


# --------------------------------------------------------------------------
# Pluggable decision weights
# --------------------------------------------------------------------------
def _tree_from_hard_weights(support, w):
    """Read off the single tree that one-hot weights select (root-down)."""
    wv = np.asarray(w)
    assignment = {}

    def expand(clade):
        if is_singleton(clade):
            return
        pidx = support.parent_clade_index[clade]
        start = support.parent_offsets[pidx]
        subsplits = support.subsplits_by_parent[pidx]
        choice = subsplits[int(np.argmax(wv[start : start + len(subsplits)]))]
        assignment[clade] = choice
        expand(choice.child1)
        expand(choice.child2)

    expand(support.root_clade)
    return support.assignment_to_parent_indices(assignment)


def _felsenstein(pi, Pn, freq, seq, n):
    child = NumpyTreeTopology(parent_indices=pi).child_indices
    part = {i: seq[:, i, :] for i in range(n)}
    for u in range(n, 2 * n - 1):
        c0, c1 = int(child[u][0]), int(child[u][1])
        part[u] = (part[c0] @ Pn.T) * (part[c1] @ Pn.T)
    return float(np.sum(np.log(part[2 * n - 2] @ freq)))


def test_exact_weights_match_default():
    s = _setup(seed=10)
    default = relaxed_log_likelihood_from_distribution(
        s["q"], s["P_clades"], s["sequences"], s["frequencies"], dag=s["dag"]
    ).numpy()
    via_fn = relaxed_log_likelihood_from_distribution(
        s["q"], s["P_clades"], s["sequences"], s["frequencies"], dag=s["dag"],
        weight_fn=exact_weights,
    ).numpy()
    np.testing.assert_allclose(default, via_fn, rtol=1e-12)


@pytest.mark.parametrize("gumbel", [True, False])
def test_hard_weights_collapse_to_single_tree(gumbel):
    """Straight-through (hard) weights => one tree's across-sites log-likelihood."""
    s = _setup(seed=11)
    support, dag = s["support"], s["dag"]
    segment_ids = tf.constant(support.segment_ids, tf.int32)
    logits = tf.convert_to_tensor(s["logits"])
    w = straight_through_weights(
        logits, segment_ids, support.parent_clade_count, temperature=0.7,
        seed=3, gumbel=gumbel,
    )
    tip = build_tip_partials(dag, s["sequences"])
    value = relaxed_log_likelihood(dag, w, s["P_clades"], tip, s["frequencies"]).numpy()

    pi = _tree_from_hard_weights(support, w)
    expected = _felsenstein(
        pi, s["P"].numpy(), s["frequencies"].numpy(), s["sequences"].numpy(), s["n"]
    )
    np.testing.assert_allclose(value, expected, rtol=1e-9)


def test_straight_through_weights_are_one_hot_in_forward():
    s = _setup(seed=12)
    support = s["support"]
    segment_ids = tf.constant(support.segment_ids, tf.int32)
    w = straight_through_weights(
        tf.convert_to_tensor(s["logits"]), segment_ids,
        support.parent_clade_count, seed=1,
    ).numpy()
    for pidx, _clade in enumerate(support.parent_clades):
        start = support.parent_offsets[pidx]
        seg = w[start : start + len(support.subsplits_by_parent[pidx])]
        np.testing.assert_allclose(seg.sum(), 1.0, atol=1e-9)
        assert np.isclose(seg.max(), 1.0, atol=1e-9)  # exactly one entry is 1


@pytest.mark.parametrize("weight_fn", [straight_through_weights, gumbel_softmax_weights])
def test_sampled_weight_gradient_nonzero(weight_fn):
    s = _setup(seed=13)
    with tf.GradientTape() as tape:
        ll = relaxed_log_likelihood_from_distribution(
            s["q"], s["P_clades"], s["sequences"], s["frequencies"], dag=s["dag"],
            weight_fn=weight_fn, temperature=0.5, seed=2,
        )
    grad = tape.gradient(ll, s["logits"]).numpy()
    assert np.all(np.isfinite(grad))
    assert np.any(grad != 0)


def test_gumbel_softmax_hardens_at_low_temperature():
    """Low temperature drives the relaxed weights toward one-hot per clade."""
    s = _setup(seed=14)
    support = s["support"]
    segment_ids = tf.constant(support.segment_ids, tf.int32)
    logits = tf.convert_to_tensor(s["logits"])
    cold = gumbel_softmax_weights(
        logits, segment_ids, support.parent_clade_count, temperature=0.01, seed=5
    ).numpy()
    maxes = []
    for pidx, _clade in enumerate(support.parent_clades):
        start = support.parent_offsets[pidx]
        seg = cold[start : start + len(support.subsplits_by_parent[pidx])]
        maxes.append(seg.max())
    assert np.mean(maxes) > 0.95
