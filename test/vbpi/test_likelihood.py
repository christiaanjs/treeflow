import numpy as np
import pytest
import tensorflow as tf

from treeflow.vbpi.clade import child_indices_from_parent_indices
from treeflow.vbpi.likelihood import (
    jc_transition_probs,
    make_jc_log_likelihood_fn,
    rooted_jc_log_likelihood,
)

from vbpi_test_helpers import random_parent_indices


def numpy_felsenstein(parent_indices, leaf_partials, branch_lengths):
    """Independent NumPy Felsenstein pruning under JC (reference)."""
    n = leaf_partials.shape[0]
    node_count = 2 * n - 1
    children = child_indices_from_parent_indices(parent_indices, node_count)

    def P(t):
        r = np.exp(-4.0 / 3.0 * t)
        return np.where(np.eye(4, dtype=bool), 0.25 + 0.75 * r, 0.25 - 0.25 * r)

    total = 0.0
    for s in range(leaf_partials.shape[1]):
        partials = {i: leaf_partials[i, s] for i in range(n)}
        for node in range(n, node_count):
            v = np.ones(4)
            for c in children[node]:
                v *= P(branch_lengths[c]) @ partials[c]
            partials[node] = v
        total += np.log(np.sum(0.25 * partials[node_count - 1]))
    return total


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    n = 6
    parent = random_parent_indices(n, rng)
    leaf = np.eye(4)[rng.integers(0, 4, size=(n, 30))]
    branch = rng.uniform(0.05, 0.6, size=2 * n - 2)
    return parent, leaf, branch


def test_jc_transition_probs_valid():
    t = tf.constant([0.0, 0.1, 1.0], dtype=tf.float64)
    P = jc_transition_probs(t).numpy()
    # rows sum to one, and t=0 gives the identity
    assert np.allclose(P.sum(-1), 1.0)
    assert np.allclose(P[0], np.eye(4))


@pytest.mark.parametrize("use_native", [False, "auto"])
def test_matches_numpy_reference(problem, use_native):
    parent, leaf, branch = problem
    ref = numpy_felsenstein(parent, leaf, branch)
    got = rooted_jc_log_likelihood(
        parent, tf.constant(leaf), tf.constant(branch), use_native=use_native
    ).numpy()
    assert np.isclose(got, ref, atol=1e-7)


def test_batched_over_branch_samples(problem):
    parent, leaf, _ = problem
    rng = np.random.default_rng(1)
    n = leaf.shape[0]
    branches = rng.uniform(0.05, 0.6, size=(4, 2 * n - 2))
    got = rooted_jc_log_likelihood(
        parent, tf.constant(leaf), tf.constant(branches), use_native=False
    ).numpy()
    ref = np.array([numpy_felsenstein(parent, leaf, branches[i]) for i in range(4)])
    assert got.shape == (4,)
    assert np.allclose(got, ref, atol=1e-7)


def test_differentiable_in_branch_lengths(problem):
    parent, leaf, branch = problem
    bl = tf.Variable(branch)
    with tf.GradientTape() as tape:
        ll = rooted_jc_log_likelihood(parent, tf.constant(leaf), bl, use_native=False)
    grad = tape.gradient(ll, bl)
    assert grad is not None
    assert np.all(np.isfinite(grad.numpy()))


def test_constant_likelihood_for_gaps():
    # All-ones (gap) partials: likelihood is 1 (log 0) for any tree/branches.
    n = 5
    rng = np.random.default_rng(2)
    parent = random_parent_indices(n, rng)
    leaf = np.ones((n, 7, 4))
    branch = rng.uniform(0.1, 0.5, size=2 * n - 2)
    ll = rooted_jc_log_likelihood(
        parent, tf.constant(leaf), tf.constant(branch), use_native=False
    ).numpy()
    assert np.isclose(ll, 0.0, atol=1e-9)


def test_function_mode_matches_eager(problem):
    from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
    from treeflow.tree.topology.tensorflow_tree_topology import (
        numpy_topology_to_tensor,
    )

    parent, leaf, branch = problem
    fn = make_jc_log_likelihood_fn(tf.constant(leaf), use_native=False)
    topology = numpy_topology_to_tensor(NumpyTreeTopology(parent_indices=parent))
    compiled = fn(topology, tf.constant(branch)).numpy()
    eager = rooted_jc_log_likelihood(
        parent, tf.constant(leaf), tf.constant(branch), use_native=False
    ).numpy()
    assert np.isclose(compiled, eager, atol=1e-10)
