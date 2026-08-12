import numpy as np
import pytest
import tensorflow as tf

from treeflow.vbpi.branch_model import SplitLognormalBranchModel
from treeflow.vbpi.sbn import SubsplitBayesianNetwork
from treeflow.vbpi.support import SubsplitSupport, all_rooted_parent_indices


@pytest.fixture
def support_and_samples():
    n = 6
    topos = all_rooted_parent_indices(n)
    support = SubsplitSupport.from_topologies(topos, n)
    sbn = SubsplitBayesianNetwork(support)
    samples = sbn.sample_topologies(16, seed=2, use_native=False)
    return support, samples


def test_sample_shapes(support_and_samples):
    support, samples = support_and_samples
    bm = SplitLognormalBranchModel(support)
    node_clade_ids = tf.constant(samples.node_clade_ids)
    branch_lengths, log_prob = bm.sample_and_log_prob(node_clade_ids, seed=(1, 2))
    n = support.taxon_count
    assert branch_lengths.shape == (16, 2 * n - 2)
    assert log_prob.shape == (16,)
    assert np.all(branch_lengths.numpy() > 0)  # log-normal is positive


def test_reparameterised_gradient(support_and_samples):
    support, samples = support_and_samples
    bm = SplitLognormalBranchModel(support)
    node_clade_ids = tf.constant(samples.node_clade_ids)
    with tf.GradientTape() as tape:
        branch_lengths, _ = bm.sample_and_log_prob(node_clade_ids, seed=(3, 4))
        # a smooth function of the (reparameterised) branch lengths
        objective = tf.reduce_sum(tf.math.log(branch_lengths))
    grads = tape.gradient(objective, [bm.loc, bm.scale_param])
    assert all(g is not None for g in grads)
    # loc shifts every branch multiplicatively, so d/dloc sum(log b) == n_branches
    n_branches = 16 * (2 * support.taxon_count - 2)
    total_loc_grad = tf.reduce_sum(grads[0]).numpy()
    assert np.isclose(total_loc_grad, n_branches, rtol=1e-6)


def test_log_prob_matches_sample(support_and_samples):
    support, samples = support_and_samples
    bm = SplitLognormalBranchModel(support)
    node_clade_ids = tf.constant(samples.node_clade_ids)
    branch_lengths, log_prob = bm.sample_and_log_prob(node_clade_ids, seed=(5, 6))
    recomputed = bm.log_prob(node_clade_ids, branch_lengths)
    assert np.allclose(log_prob.numpy(), recomputed.numpy())


def test_init_scale_respected(support_and_samples):
    support, _ = support_and_samples
    bm = SplitLognormalBranchModel(support, init_loc=-1.5, init_scale=0.25)
    assert np.allclose(bm.scale().numpy(), 0.25, atol=1e-6)
    assert np.allclose(bm.loc.numpy(), -1.5)


def test_shared_split_shares_parameters():
    # Two topologies sharing a clade edge must use the same branch parameter.
    n = 5
    topos = all_rooted_parent_indices(n)
    support = SubsplitSupport.from_topologies(topos, n)
    bm = SplitLognormalBranchModel(support)
    # leaf clade {0} always exists; its branch parameter index is its clade id
    leaf0 = support.clade_id(frozenset({0}))
    # Set a distinctive loc for that clade
    loc = bm.loc.numpy()
    loc[leaf0] = 3.0
    bm.loc.assign(loc)
    sbn = SubsplitBayesianNetwork(support)
    samples = sbn.sample_topologies(4, seed=0, use_native=False)
    node_clade_ids = tf.constant(samples.node_clade_ids)
    dist_loc = tf.gather(
        bm.loc, bm._branch_clade_ids(node_clade_ids)
    ).numpy()
    # taxon 0's branch (node id 0) must carry loc 3.0 in every sample
    assert np.allclose(dist_loc[:, 0], 3.0)
