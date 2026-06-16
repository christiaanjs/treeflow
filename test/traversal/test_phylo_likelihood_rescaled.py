"""Tests for the rescaled (numerically stable) TensorFlow likelihood and the
rescaled/unrescaled dispatcher."""
import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
from treeflow.tree.topology.tensorflow_tree_topology import numpy_topology_to_tensor
from treeflow.traversal.phylo_likelihood import (
    phylogenetic_likelihood,
    phylogenetic_log_likelihood_rescaled,
)
from treeflow.traversal.phylo_likelihood_dispatch import (
    phylogenetic_log_likelihood,
    default_rescaling_threshold,
)


def _make_problem(leaf_count, state_count, site_count, seed=0, dtype=tf.float64):
    rng = np.random.default_rng(seed)
    node_count = 2 * leaf_count - 1
    parent = np.full(node_count, -1, dtype=np.int32)
    active = list(range(leaf_count))
    nxt = leaf_count
    while len(active) > 1:
        a = active.pop(rng.integers(len(active)))
        b = active.pop(rng.integers(len(active)))
        parent[a] = parent[b] = nxt
        active.append(nxt)
        nxt += 1
    topology = numpy_topology_to_tensor(NumpyTreeTopology(parent_indices=parent[:-1]))
    np_dtype = dtype.as_numpy_dtype
    states = rng.integers(0, state_count, size=(site_count, leaf_count))
    sequences = tf.constant(np.eye(state_count, dtype=np_dtype)[states])
    probs = rng.uniform(0.1, 1.0, size=(1, node_count, state_count, state_count))
    probs = tf.constant((probs / probs.sum(-1, keepdims=True)).astype(np_dtype))
    freqs = tf.constant(np.full(state_count, 1.0 / state_count, dtype=np_dtype))
    return dict(
        topology=topology,
        sequences=sequences,
        transition_probs=probs,
        frequencies=freqs,
    )


@pytest.fixture
def small():
    return _make_problem(8, 4, 16, seed=2)


def _args(p):
    return (p["topology"], p["sequences"], p["transition_probs"], p["frequencies"])


@pytest.mark.parametrize("function_mode", [False, True])
def test_tf_rescaled_matches_unrescaled_log(small, function_mode):
    fn = (
        tf.function(phylogenetic_log_likelihood_rescaled)
        if function_mode
        else phylogenetic_log_likelihood_rescaled
    )
    batch = tf.shape(small["sequences"])[:1]
    rescaled = fn(*_args(small), batch_shape=batch)
    unrescaled = phylogenetic_likelihood(*_args(small), batch_shape=batch)
    assert_allclose(rescaled.numpy(), tf.math.log(unrescaled).numpy(),
                    rtol=1e-12, atol=1e-12)


def test_tf_rescaled_gradient_matches(small):
    batch = tf.shape(small["sequences"])[:1]

    def grads(rescaled):
        probs = tf.Variable(small["transition_probs"])
        with tf.GradientTape() as tape:
            if rescaled:
                ll = phylogenetic_log_likelihood_rescaled(
                    small["topology"], small["sequences"], probs,
                    small["frequencies"], batch_shape=batch)
            else:
                ll = tf.math.log(phylogenetic_likelihood(
                    small["topology"], small["sequences"], probs,
                    small["frequencies"], batch_shape=batch))
            loss = tf.reduce_sum(ll)
        return tape.gradient(loss, probs)

    assert_allclose(grads(True).numpy(), grads(False).numpy(), rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("unroll", ["unrolled", "tensorarray", "while_loop"])
def test_tf_rescaled_avoids_underflow(unroll):
    # 600 taxa exercises the likelihood traversal well past the profiler's sizes;
    # parametrising over the traversal modes checks each scales.
    p = _make_problem(600, 4, 4, seed=5)
    batch = tf.shape(p["sequences"])[:1]
    unrescaled = phylogenetic_likelihood(*_args(p), batch_shape=batch, unroll=unroll)
    rescaled = phylogenetic_log_likelihood_rescaled(
        *_args(p), batch_shape=batch, unroll=unroll
    )
    assert np.all(unrescaled.numpy() == 0.0)
    assert np.all(np.isfinite(rescaled.numpy()))

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_default_threshold_dtype_dependent():
    t64 = default_rescaling_threshold(tf.float64)
    t32 = default_rescaling_threshold(tf.float32)
    assert t64 > t32 > 0


@pytest.mark.parametrize("rescaling", [False, True, "auto", "adaptive"])
def test_dispatch_modes_agree_small(small, rescaling):
    batch = tf.shape(small["sequences"])[:1]
    expected = tf.math.log(phylogenetic_likelihood(*_args(small), batch_shape=batch))
    got = phylogenetic_log_likelihood(*_args(small), batch_shape=batch,
                                      rescaling=rescaling)
    assert_allclose(got.numpy(), expected.numpy(), rtol=1e-11, atol=1e-11)


def test_dispatch_auto_picks_rescaled_for_large_tree():
    # Force a low threshold so the small tree triggers the rescaled path.
    p = _make_problem(8, 4, 8, seed=1)
    batch = tf.shape(p["sequences"])[:1]
    got = phylogenetic_log_likelihood(*_args(p), batch_shape=batch,
                                      rescaling="auto", rescaling_threshold=2)
    expected = phylogenetic_log_likelihood_rescaled(*_args(p), batch_shape=batch)
    assert_allclose(got.numpy(), expected.numpy(), rtol=1e-12, atol=1e-12)


def test_dispatch_adaptive_falls_back_on_underflow():
    p = _make_problem(600, 4, 4, seed=7)
    batch = tf.shape(p["sequences"])[:1]
    got = phylogenetic_log_likelihood(*_args(p), batch_shape=batch,
                                      rescaling="adaptive")
    assert np.all(np.isfinite(got.numpy()))


def test_dispatch_invalid_mode_raises(small):
    with pytest.raises(ValueError):
        phylogenetic_log_likelihood(*_args(small), rescaling="nonsense")
