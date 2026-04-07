"""
Unit and integration tests for the CPP birth-death tree sampler.
"""
import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.distributions.tree.birthdeath.cpp_sampler import (
    build_random_topologies,
    sample_bd_tree,
    sample_origin_age,
    sample_speciation_times,
)
from treeflow.distributions.tree.birthdeath.birth_death_contemporary_sampling import (
    BirthDeathContemporarySampling,
)
from treeflow.distributions.tree.birthdeath.yule import Yule

SEED = tf.constant([42, 7], dtype=tf.int32)
DTYPE = DEFAULT_FLOAT_DTYPE_TF


# ── helpers ──────────────────────────────────────────────────────────────────

def _c(v):
    return tf.constant(v, dtype=DTYPE)


# ── Stage 1: sample_origin_age ────────────────────────────────────────────────

def test_sample_origin_age_shape_scalar():
    T = sample_origin_age(5, _c(2.0), _c(1.0), _c(0.5), 100, SEED)
    assert T.shape == (100,), f"Expected (100,), got {T.shape}"


def test_sample_origin_age_shape_batched():
    lambda_ = _c([2.0, 3.0])
    mu = _c([1.0, 1.5])
    rho = _c([0.5, 0.8])
    T = sample_origin_age(5, lambda_, mu, rho, 50, SEED)
    assert T.shape == (50, 2), f"Expected (50, 2), got {T.shape}"


def test_sample_origin_age_positive():
    T = sample_origin_age(5, _c(2.0), _c(1.0), _c(0.5), 200, SEED)
    assert tf.reduce_all(T > 0).numpy(), "All origin ages should be positive"


def test_sample_origin_age_yule():
    """Yule (mu=0, rho=1) case: T should be positive."""
    T = sample_origin_age(5, _c(1.0), _c(0.0), _c(1.0), 200, SEED)
    assert tf.reduce_all(T > 0).numpy()


def test_sample_origin_age_reproducible():
    T1 = sample_origin_age(4, _c(1.5), _c(0.5), _c(1.0), 10, SEED)
    T2 = sample_origin_age(4, _c(1.5), _c(0.5), _c(1.0), 10, SEED)
    assert_allclose(T1.numpy(), T2.numpy())


# ── Stage 2: sample_speciation_times ─────────────────────────────────────────

def test_sample_speciation_times_shape():
    T = tf.constant([0.5, 1.0, 0.8], dtype=DTYPE)  # 3 origin ages
    tau = sample_speciation_times(T, 4, _c(2.0), _c(1.0), _c(0.5), SEED)
    assert tau.shape == (3, 3), f"Expected (3, 3), got {tau.shape}"


def test_sample_speciation_times_shape_batched():
    # T: [n_samples, batch]  lambda_: [batch]
    lambda_ = _c([2.0, 3.0])
    mu = _c([1.0, 1.5])
    rho = _c([0.5, 0.8])
    T = sample_origin_age(5, lambda_, mu, rho, 20, SEED)  # [20, 2]
    tau = sample_speciation_times(T, 5, lambda_, mu, rho, SEED)
    assert tau.shape == (20, 2, 4), f"Expected (20, 2, 4), got {tau.shape}"


def test_sample_speciation_times_sorted():
    T = _c([1.0] * 50)  # fixed origin age
    tau = sample_speciation_times(T, 5, _c(2.0), _c(1.0), _c(0.5), SEED)
    # Each row should be sorted ascending
    diffs = tau[:, 1:] - tau[:, :-1]
    assert tf.reduce_all(diffs >= 0).numpy(), "Speciation times should be sorted ascending"


def test_sample_speciation_times_in_range():
    T_val = 1.0
    T = _c([T_val] * 100)
    tau = sample_speciation_times(T, 4, _c(2.0), _c(1.0), _c(0.5), SEED)
    assert tf.reduce_all(tau >= 0).numpy(), "Speciation times must be >= 0"
    assert tf.reduce_all(tau <= T_val).numpy(), f"Speciation times must be <= T={T_val}"


def test_sample_speciation_times_n2():
    """n_taxa=2: exactly 1 speciation time per sample."""
    T = _c([0.5, 0.8])
    tau = sample_speciation_times(T, 2, _c(2.0), _c(1.0), _c(1.0), SEED)
    assert tau.shape == (2, 1), f"Expected (2, 1), got {tau.shape}"


# ── Stage 3: build_random_topologies ─────────────────────────────────────────

def test_build_random_topologies_shapes():
    n_taxa = 5
    n = 2 * n_taxa - 1
    topo = build_random_topologies(30, n_taxa, SEED)
    assert topo.parent_indices.shape == (30, 2 * n_taxa - 2)
    assert topo.child_indices.shape == (30, n, 2)
    assert topo.preorder_indices.shape == (30, n)


def test_build_random_topologies_root():
    """Root (last internal node) should have no parent in parent_indices."""
    n_taxa = 5
    topo = build_random_topologies(20, n_taxa, SEED)
    pi = topo.parent_indices.numpy()  # [20, 2n-2]
    root = 2 * n_taxa - 2  # root index
    # All internal nodes except root should appear somewhere as a parent
    # The root itself (index 2n-2) is NOT in parent_indices (no row for it)
    assert pi.shape[-1] == root, "parent_indices has 2n-2 entries (all non-root nodes)"


def test_build_random_topologies_each_node_has_two_children():
    """Every internal node must have exactly 2 valid children."""
    n_taxa = 5
    topo = build_random_topologies(20, n_taxa, SEED)
    ci = topo.child_indices.numpy()  # [20, 2n-1, 2]
    for s in range(20):
        for node in range(n_taxa, 2 * n_taxa - 1):
            children = ci[s, node]
            assert children[0] >= 0 and children[1] >= 0, (
                f"sample {s}, internal node {node} has invalid child: {children}"
            )
            assert children[0] != children[1], (
                f"sample {s}, internal node {node} has identical children"
            )


def test_build_random_topologies_leaves_have_no_children():
    """Leaf nodes should have child_indices == -1."""
    n_taxa = 5
    topo = build_random_topologies(10, n_taxa, SEED)
    ci = topo.child_indices.numpy()
    for leaf in range(n_taxa):
        assert np.all(ci[:, leaf, :] == -1), f"Leaf {leaf} should have no children"


def test_build_random_topologies_parent_consistency():
    """For each internal node, its children's parent_indices should point back."""
    n_taxa = 5
    topo = build_random_topologies(20, n_taxa, SEED)
    pi = topo.parent_indices.numpy()  # [20, 2n-2]: pi[s, i] = parent of node i
    ci = topo.child_indices.numpy()   # [20, 2n-1, 2]
    for s in range(20):
        for internal in range(n_taxa, 2 * n_taxa - 1):
            for child in ci[s, internal]:
                assert pi[s, child] == internal, (
                    f"sample {s}: child {child} of node {internal} "
                    f"has parent {pi[s, child]}"
                )


def test_build_random_topologies_preorder_root_first():
    """Root should be the first element of preorder_indices."""
    n_taxa = 5
    root = 2 * n_taxa - 2
    topo = build_random_topologies(20, n_taxa, SEED)
    pre = topo.preorder_indices.numpy()
    assert np.all(pre[:, 0] == root), "Root must be first in preorder"


def test_build_random_topologies_reproducible():
    topo1 = build_random_topologies(5, 4, SEED)
    topo2 = build_random_topologies(5, 4, SEED)
    assert_allclose(topo1.parent_indices.numpy(), topo2.parent_indices.numpy())


def test_build_random_topologies_n2():
    """Minimal tree with 2 taxa: single internal node = root = node 2."""
    topo = build_random_topologies(10, 2, SEED)
    pi = topo.parent_indices.numpy()  # [10, 2]: nodes 0 and 1 both point to 2
    assert np.all(pi == 2), f"Both leaves should have parent 2, got:\n{pi}"


# ── build_random_topologies: preorder / traversal validity ───────────────────

def test_build_random_topologies_preorder_contains_all_nodes():
    """preorder_indices contains every node 0..2n-1 exactly once."""
    n_taxa = 5
    node_count = 2 * n_taxa - 1
    topo = build_random_topologies(20, n_taxa, SEED)
    pre = topo.preorder_indices.numpy()
    for s in range(20):
        assert sorted(pre[s].tolist()) == list(range(node_count)), (
            f"sample {s}: preorder does not contain all nodes"
        )


def test_build_random_topologies_preorder_parent_before_child():
    """In preorder_indices every node's parent must appear at an earlier position."""
    n_taxa = 5
    topo = build_random_topologies(20, n_taxa, SEED)
    pi = topo.parent_indices.numpy()   # [20, 2n-2]
    pre = topo.preorder_indices.numpy()  # [20, 2n-1]
    for s in range(20):
        position = {int(node): pos for pos, node in enumerate(pre[s])}
        for node in range(2 * n_taxa - 2):  # all non-root nodes
            parent = int(pi[s, node])
            assert position[parent] < position[node], (
                f"sample {s}: parent {parent} (pos {position[parent]}) "
                f"must precede node {node} (pos {position[node]})"
            )


def test_build_random_topologies_preorder_same_across_samples():
    """preorder_indices is identical for every sample in the batch."""
    n_taxa = 5
    topo = build_random_topologies(20, n_taxa, SEED)
    pre = topo.preorder_indices.numpy()
    for s in range(1, 20):
        np.testing.assert_array_equal(
            pre[s], pre[0],
            err_msg=f"sample {s} preorder differs from sample 0",
        )


def test_build_random_topologies_preorder_works_with_ratio_transform():
    """Sampled topology preorder indices are compatible with ratios_to_node_heights."""
    from treeflow.traversal.ratio_transform import ratios_to_node_heights

    n_taxa = 5
    n_internal = n_taxa - 1
    topo = build_random_topologies(1, n_taxa, SEED)

    # Squeeze the batch dim and work with raw tensors to avoid the batched
    # restriction in the preorder_node_indices property.
    pi = topo.parent_indices[0]             # [2n-2]
    preorder = topo.preorder_indices[0]     # [2n-1]

    # Filter to internal nodes (global index >= n_taxa) and shift to local space
    mask = preorder >= n_taxa
    pre_node_local = tf.boolean_mask(preorder, mask) - n_taxa   # [n-1]
    pi_internal_local = pi[n_taxa:] - n_taxa                     # [n-2] (non-root only)

    ratios = tf.fill([n_internal], tf.constant(0.5, dtype=tf.float32))
    anchor = tf.zeros([n_internal], dtype=tf.float32)

    heights = ratios_to_node_heights(pre_node_local, pi_internal_local, ratios, anchor)
    assert heights.shape == (n_internal,), f"Unexpected shape {heights.shape}"
    assert tf.reduce_all(tf.math.is_finite(heights)).numpy(), (
        f"heights contains non-finite values: {heights.numpy()}"
    )


# ── Integration: BirthDeathContemporarySampling.sample ───────────────────────

def test_bd_sample_shape():
    dist = BirthDeathContemporarySampling(
        5, _c(1.0), _c(0.5), sample_probability=_c(1.0)
    )
    samples = dist.sample(3, seed=SEED)
    assert samples.node_heights.shape == (3, 4)
    assert samples.sampling_times.shape == (3, 5)
    assert samples.topology.parent_indices.shape == (3, 8)


def test_bd_sample_log_prob_finite():
    """log_prob evaluated on sampled trees should be finite."""
    dist = BirthDeathContemporarySampling(
        5, _c(1.0), _c(0.5), sample_probability=_c(1.0)
    )
    samples = dist.sample(10, seed=SEED)
    lp = dist.log_prob(samples)
    assert lp.shape == (10,), f"Expected (10,), got {lp.shape}"
    assert tf.reduce_all(tf.math.is_finite(lp)).numpy(), (
        f"log_prob has non-finite values: {lp.numpy()}"
    )


def test_bd_sample_node_heights_positive():
    dist = BirthDeathContemporarySampling(5, _c(1.0), _c(0.5))
    samples = dist.sample(20, seed=SEED)
    assert tf.reduce_all(samples.node_heights > 0).numpy()


def test_bd_sample_heights_sorted():
    """Node heights must be non-decreasing along the last axis (postorder)."""
    dist = BirthDeathContemporarySampling(5, _c(1.0), _c(0.5))
    samples = dist.sample(20, seed=SEED)
    h = samples.node_heights
    diffs = h[:, 1:] - h[:, :-1]
    assert tf.reduce_all(diffs >= 0).numpy(), "Node heights should be sorted ascending"


def test_bd_sample_batch_shape():
    """Batched parameters: samples have matching batch shape."""
    lambda_vals = _c([1.0, 2.0])
    a_vals = _c([0.3, 0.5])
    dist = BirthDeathContemporarySampling(4, lambda_vals, a_vals)
    samples = dist.sample(5, seed=SEED)
    assert samples.node_heights.shape == (5, 2, 3)
    assert samples.topology.parent_indices.shape == (5, 2, 6)


def test_bd_sample_batch_log_prob_finite():
    lambda_vals = _c([1.0, 2.0])
    a_vals = _c([0.3, 0.5])
    dist = BirthDeathContemporarySampling(4, lambda_vals, a_vals)
    samples = dist.sample(5, seed=SEED)
    lp = dist.log_prob(samples)
    assert lp.shape == (5, 2), f"Expected (5, 2), got {lp.shape}"
    assert tf.reduce_all(tf.math.is_finite(lp)).numpy()


# ── Yule subclass ─────────────────────────────────────────────────────────────

def test_yule_sample_shape():
    dist = Yule(5, _c(2.0))
    samples = dist.sample(10, seed=SEED)
    assert samples.node_heights.shape == (10, 4)


def test_yule_sample_log_prob_finite():
    dist = Yule(5, _c(2.0))
    samples = dist.sample(10, seed=SEED)
    lp = dist.log_prob(samples)
    assert tf.reduce_all(tf.math.is_finite(lp)).numpy(), (
        f"Yule log_prob has non-finite values: {lp.numpy()}"
    )
