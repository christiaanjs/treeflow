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
    sample_ranking,
    sample_speciation_times,
)
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
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


# ── sample_ranking ───────────────────────────────────────────────────────────

# Caterpillar n_taxa=4: 4->{0,1}, 5->{4,2}, 6->{5,3}
# parent_indices[0..5] = [4,4,5,6,5,6]  (parents of nodes 0..5; root 6 absent)
# child_indices[4]=[0,1], [5]=[2,4], [6]=[3,5]  (sorted min-first)
_CAT4_PARENT = tf.constant([4, 4, 5, 6, 5, 6], dtype=tf.int32)
_CAT4_CHILD = tf.constant(
    [[-1,-1],[-1,-1],[-1,-1],[-1,-1],[0,1],[2,4],[3,5]], dtype=tf.int32
)
# Only one valid ranking: local 0 (node 4) < local 1 (node 5) < local 2 (node 6)

# Balanced n_taxa=4: 4->{0,1}, 5->{2,3}, 6->{4,5}
# parent_indices[0..5] = [4,4,5,5,6,6]
# child_indices[4]=[0,1], [5]=[2,3], [6]=[4,5]
_BAL4_PARENT = tf.constant([4, 4, 5, 5, 6, 6], dtype=tf.int32)
_BAL4_CHILD = tf.constant(
    [[-1,-1],[-1,-1],[-1,-1],[-1,-1],[0,1],[2,3],[4,5]], dtype=tf.int32
)
# Two valid rankings: [0,1,2] and [1,0,2]  (node 4 or node 5 can go first)


def test_sample_ranking_is_permutation():
    """sigma[s, :] contains every local index 0..n_internal-1 exactly once."""
    n_taxa = 5
    topo = build_random_topologies(1, n_taxa, SEED)
    ci = topo.child_indices[0]   # [2n-1, 2]
    pi = topo.parent_indices[0]  # [2n-2]
    n_internal = n_taxa - 1
    sigma = sample_ranking(n_taxa, ci, pi, 20, SEED)
    for s in range(20):
        assert sorted(sigma[s].numpy().tolist()) == list(range(n_internal)), (
            f"sample {s}: sigma not a permutation: {sigma[s].numpy()}"
        )


def test_sample_ranking_parent_has_higher_rank():
    """For caterpillar n_taxa=4, only one valid ranking — verify it is always produced."""
    n_taxa = 4
    n_samples = 50
    sigma = sample_ranking(n_taxa, _CAT4_CHILD, _CAT4_PARENT, n_samples, SEED)
    # sigma_inv[s, local_node] = rank of that node
    # Build inverse: sigma_inv[s, sigma[s, k]] = k
    sigma_np = sigma.numpy()
    n_internal = n_taxa - 1
    for s in range(n_samples):
        sigma_inv = np.empty(n_internal, dtype=int)
        sigma_inv[sigma_np[s]] = np.arange(n_internal)
        # parent local indices: node 0→parent1(local 1), node 1→parent2(local 2)
        # local 0: parent is local 1; local 1: parent is local 2 (root)
        assert sigma_inv[1] > sigma_inv[0], f"sample {s}: parent rank <= child rank"
        assert sigma_inv[2] > sigma_inv[1], f"sample {s}: root rank <= child rank"


def test_sample_ranking_uniform():
    """For balanced n_taxa=4, the two valid rankings should each occur ~50%."""
    n_taxa = 4
    n_samples = 2000
    sigma = sample_ranking(n_taxa, _BAL4_CHILD, _BAL4_PARENT, n_samples, SEED)
    sigma_np = sigma.numpy()
    # Valid rankings in local space: [0,1,2] (node 4 first) or [1,0,2] (node 5 first)
    count_01 = np.sum((sigma_np[:, 0] == 0) & (sigma_np[:, 1] == 1))
    count_10 = np.sum((sigma_np[:, 0] == 1) & (sigma_np[:, 1] == 0))
    assert count_01 + count_10 == n_samples, "Unexpected ranking produced"
    # Each should be ~50%; allow 5-sigma tolerance for binomial(2000, 0.5)
    expected = n_samples / 2
    tol = 5 * (n_samples * 0.25) ** 0.5  # 5 * std
    assert abs(count_01 - expected) < tol, (
        f"Ranking [0,1,2] count {count_01} too far from {expected}"
    )


def test_sample_ranking_reproducible():
    """Same seed produces identical sigma."""
    n_taxa = 5
    topo = build_random_topologies(1, n_taxa, SEED)
    ci, pi = topo.child_indices[0], topo.parent_indices[0]
    s1 = sample_ranking(n_taxa, ci, pi, 10, SEED)
    s2 = sample_ranking(n_taxa, ci, pi, 10, SEED)
    assert_allclose(s1.numpy(), s2.numpy())


# ── BirthDeathContemporarySampling with fixed topology ───────────────────────

def _make_fixed_topology(n_taxa):
    """Return a balanced-ish fixed TensorflowTreeTopology for n_taxa."""
    topo = build_random_topologies(1, n_taxa, SEED)
    return TensorflowTreeTopology(
        parent_indices=topo.parent_indices[0],
        child_indices=topo.child_indices[0],
        preorder_indices=topo.preorder_indices[0],
    )


def test_bd_fixed_topology_shape():
    n_taxa = 5
    fixed_topo = _make_fixed_topology(n_taxa)
    dist = BirthDeathContemporarySampling(
        n_taxa, _c(1.0), _c(0.5), sample_probability=_c(1.0),
        fixed_topology=fixed_topo,
    )
    samples = dist.sample(4, seed=SEED)
    assert samples.node_heights.shape == (4, n_taxa - 1)
    assert samples.topology.parent_indices.shape == (4, 2 * n_taxa - 2)


def test_bd_fixed_topology_parent_height_gt_child_height():
    """For every internal-node parent/child pair the parent height must exceed the child height."""
    n_taxa = 5
    fixed_topo = _make_fixed_topology(n_taxa)
    dist = BirthDeathContemporarySampling(
        n_taxa, _c(1.0), _c(0.5), sample_probability=_c(1.0),
        fixed_topology=fixed_topo,
    )
    samples = dist.sample(20, seed=SEED)
    # node_heights[s, j] = height of internal node n_taxa+j (local index j)
    heights = samples.node_heights.numpy()          # [20, n_taxa-1]
    pi = fixed_topo.parent_indices.numpy()          # [2n-2]
    n_internal = n_taxa - 1
    for j in range(n_internal - 1):     # root (local n_internal-1) has no parent entry
        parent_global = pi[n_taxa + j]  # global parent of internal node n_taxa+j
        parent_local = parent_global - n_taxa
        assert np.all(heights[:, parent_local] > heights[:, j]), (
            f"Parent local {parent_local} height not always > child local {j}"
        )


def test_bd_fixed_topology_log_prob_finite():
    n_taxa = 5
    fixed_topo = _make_fixed_topology(n_taxa)
    dist = BirthDeathContemporarySampling(
        n_taxa, _c(1.0), _c(0.5), sample_probability=_c(1.0),
        fixed_topology=fixed_topo,
    )
    samples = dist.sample(10, seed=SEED)
    lp = dist.log_prob(samples)
    assert lp.shape == (10,)
    assert tf.reduce_all(tf.math.is_finite(lp)).numpy(), (
        f"log_prob has non-finite values: {lp.numpy()}"
    )


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
