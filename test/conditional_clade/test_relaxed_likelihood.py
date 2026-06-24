"""Tests for the selection-differentiable phylogenetic likelihood.

The efficient ``straight_through_gather`` path (hard gather forward, one-hot-
multiply gradient) must agree with autodiff of the dense one-hot multiply, and the
relaxed likelihood must equal a direct Felsenstein computation.
"""

import numpy as np
import pytest
import tensorflow as tf

from treeflow.conditional_clade.distribution import ConditionalCladeDistribution
from treeflow.conditional_clade.relaxed_likelihood import (
    child_selection_from_topology,
    relaxed_phylogenetic_likelihood,
    straight_through_gather,
)
from treeflow.conditional_clade.support import ConditionalCladeSupport
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
from treeflow.tree.topology.tensorflow_tree_topology import numpy_topology_to_tensor


def _problem(n=5, state=4, sites=6, seed=0):
    rng = np.random.default_rng(seed)
    support = ConditionalCladeSupport(n)
    ccd = ConditionalCladeDistribution(
        support, tf.constant(rng.standard_normal(support.subsplit_count))
    )
    parent_indices = ccd.sample_parent_indices(rng)
    topology = numpy_topology_to_tensor(
        NumpyTreeTopology(parent_indices=parent_indices)
    )
    node_count = 2 * n - 1
    transition = rng.dirichlet(np.ones(state), size=(node_count, state))
    frequencies = np.ones(state) / state
    sequences = np.eye(state)[rng.integers(0, state, size=(sites, n))]
    return dict(
        n=n, state=state, sites=sites, node_count=node_count,
        parent_indices=parent_indices, topology=topology,
        transition=transition, frequencies=frequencies, sequences=sequences,
    )


def _manual_felsenstein(problem):
    n, node_count = problem["n"], problem["node_count"]
    trans, freq, seq = (
        problem["transition"], problem["frequencies"], problem["sequences"]
    )
    child = NumpyTreeTopology(parent_indices=problem["parent_indices"]).child_indices
    partials = {i: seq[:, i, :] for i in range(n)}
    for u in range(n, node_count):
        c0, c1 = child[u]
        partials[u] = (partials[c0] @ trans[c0].T) * (partials[c1] @ trans[c1].T)
    return partials[node_count - 1] @ freq


def _relaxed_args(problem):
    n, node_count, sites, state = (
        problem["n"], problem["node_count"], problem["sites"], problem["state"]
    )
    trans_relaxed = tf.constant(
        np.transpose(
            np.broadcast_to(problem["transition"], (sites, node_count, state, state)),
            (1, 0, 2, 3),
        )
    )
    return (
        tf.constant(problem["sequences"]),
        trans_relaxed,
        tf.constant(problem["frequencies"]),
    )


def test_straight_through_gather_matches_matmul():
    rng = np.random.default_rng(1)
    num_candidates, sites, state = 9, 6, 4
    values = tf.constant(rng.standard_normal((num_candidates, sites, state)))
    one_hot = np.zeros((2, num_candidates))
    one_hot[0, 3] = 1.0
    one_hot[1, 7] = 1.0
    selection = tf.Variable(tf.constant(one_hot))

    with tf.GradientTape() as tape:
        gathered = straight_through_gather(values, selection)
        loss = tf.reduce_sum(gathered ** 2)
    grad_gather = tape.gradient(loss, selection)

    with tf.GradientTape() as tape:
        product = tf.tensordot(selection, values, axes=[[1], [0]])
        loss = tf.reduce_sum(product ** 2)
    grad_matmul = tape.gradient(loss, selection)

    np.testing.assert_allclose(gathered.numpy(), product.numpy())
    np.testing.assert_allclose(grad_gather.numpy(), grad_matmul.numpy())


@pytest.mark.parametrize("n", [4, 5, 6])
def test_relaxed_dense_matches_manual_felsenstein(n):
    problem = _problem(n=n, seed=n)
    selection = child_selection_from_topology(problem["topology"], tf.float64)
    seq, trans, freq = _relaxed_args(problem)
    dense = relaxed_phylogenetic_likelihood(
        selection, seq, trans, freq, n, gather=False
    ).numpy()
    np.testing.assert_allclose(dense, _manual_felsenstein(problem))


@pytest.mark.parametrize("n", [4, 5, 6])
def test_gather_path_matches_dense_forward_and_gradient(n):
    problem = _problem(n=n, seed=n + 10)
    selection = tf.Variable(child_selection_from_topology(problem["topology"], tf.float64))
    seq, trans, freq = _relaxed_args(problem)

    def log_likelihood(gather):
        return tf.reduce_sum(
            tf.math.log(
                relaxed_phylogenetic_likelihood(
                    selection, seq, trans, freq, n, gather=gather
                )
            )
        )

    with tf.GradientTape() as tape:
        dense = log_likelihood(False)
    grad_dense = tape.gradient(dense, selection)
    with tf.GradientTape() as tape:
        gathered = log_likelihood(True)
    grad_gather = tape.gradient(gathered, selection)

    np.testing.assert_allclose(float(dense), float(gathered))
    np.testing.assert_allclose(grad_dense.numpy(), grad_gather.numpy())


def test_gather_forward_is_a_hard_gather():
    """The efficient path's forward pass is a plain gather (no dense one-hot)."""
    problem = _problem(n=5, seed=3)
    # A non-one-hot selection still gathers by argmax on the forward pass.
    selection = child_selection_from_topology(problem["topology"], tf.float64)
    perturbed = selection + 0.01  # argmax unchanged, but a matmul would differ
    seq, trans, freq = _relaxed_args(problem)
    gathered = relaxed_phylogenetic_likelihood(
        perturbed, seq, trans, freq, 5, gather=True
    ).numpy()
    exact = relaxed_phylogenetic_likelihood(
        selection, seq, trans, freq, 5, gather=True
    ).numpy()
    np.testing.assert_allclose(gathered, exact)
