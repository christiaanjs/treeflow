import pytest
import tensorflow as tf
from tensorflow_probability.python.distributions import Sample
import numpy as np
from numpy.testing import assert_allclose
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.tree.unrooted.tensorflow_unrooted_tree import TensorflowUnrootedTree
from treeflow.tree.rooted.tensorflow_rooted_tree import (
    TensorflowRootedTree,
    convert_tree_to_tensor,
)
from treeflow.tree.topology.tensorflow_tree_topology import (
    numpy_topology_to_tensor,
    TensorflowTreeTopology,
)
from treeflow.tree.topology.numpy_topology_operations import (
    get_child_indices,
    get_preorder_indices,
)
from treeflow.evolution.substitution.nucleotide.hky import HKY
from treeflow.distributions.leaf_ctmc import LeafCTMC
from treeflow.evolution.seqio import Alignment
from treeflow.evolution.substitution.probabilities import (
    get_transition_probabilities_tree,
)


@pytest.fixture
def transition_prob_tree(flat_tree_test_data):
    tree = convert_tree_to_tensor(
        NumpyRootedTree(
            node_heights=flat_tree_test_data.node_heights,
            sampling_times=flat_tree_test_data.sampling_times,
            parent_indices=flat_tree_test_data.parent_indices,
        )
    ).get_unrooted_tree()
    state_count = 5
    transition_probs = tf.fill(
        tree.branch_lengths.shape + (state_count, state_count), 1.0 / state_count
    )
    return TensorflowUnrootedTree(
        branch_lengths=transition_probs,
        topology=numpy_topology_to_tensor(tree.topology),
    )


@pytest.mark.parametrize("with_rate_categories", [True, False])
def test_LeafCTMC_batch_shape(
    tree_test_data, flat_tree_test_data, with_rate_categories
):
    tree = convert_tree_to_tensor(
        NumpyRootedTree(
            node_heights=tree_test_data.node_heights,
            sampling_times=tree_test_data.sampling_times,
            parent_indices=flat_tree_test_data.parent_indices,
        )
    ).get_unrooted_tree()
    state_count = 5
    rate_category_shape = (3,) if with_rate_categories else ()
    transition_probs = tf.fill(
        tree.branch_lengths.shape[:-1]
        + rate_category_shape
        + tree.branch_lengths.shape[-1:]
        + (state_count, state_count),
        1.0 / state_count,
    )
    transition_probs_tree = TensorflowUnrootedTree(
        branch_lengths=transition_probs,
        topology=numpy_topology_to_tensor(tree.topology),
    )
    batch_shape = tree_test_data.node_heights.shape[:-1]
    frequencies = tf.fill(
        batch_shape + (state_count,), tf.constant(0.25, dtype=transition_probs.dtype)
    )
    if rate_category_shape:
        frequencies = tf.expand_dims(frequencies, -2)
    dist = LeafCTMC(transition_probs_tree, frequencies)
    res = dist.batch_shape
    expected = batch_shape + rate_category_shape
    assert res == expected


@pytest.mark.parametrize("function_mode", [True, False])
def test_LeafCTMC_event_shape(transition_prob_tree, hky_params, function_mode):
    state_count = transition_prob_tree.branch_lengths.numpy().shape[-1]

    def event_shape_function(transition_prob_tree, frequencies):
        dist = LeafCTMC(transition_prob_tree, frequencies)
        return dist.event_shape

    if function_mode:
        event_shape_function = tf.function(event_shape_function)

    res = event_shape_function(transition_prob_tree, hky_params["frequencies"])
    if function_mode:
        res = res.numpy()
    assert tuple(res) == (transition_prob_tree.taxon_count, state_count)


def test_LeafCTMC_event_shape_tensor(transition_prob_tree, hky_params):
    state_count = transition_prob_tree.branch_lengths.numpy().shape[-1]
    dist = LeafCTMC(transition_prob_tree, hky_params["frequencies"])
    assert tuple(dist.event_shape_tensor().numpy()) == (
        transition_prob_tree.taxon_count,
        state_count,
    )


@pytest.mark.parametrize("function_mode", [True, False])
def test_LeafCTMC_log_prob_over_sites(
    hello_tensor_tree: TensorflowRootedTree,
    hello_alignment: Alignment,
    hky_params,
    hello_hky_log_likelihood,
    function_mode: bool,
):
    """Integration-style test"""

    def log_prob_fn(tree, sequences):
        transition_prob_tree = get_transition_probabilities_tree(
            tree.get_unrooted_tree(), HKY(), **hky_params
        )  # TODO: Better solution for batch dimensions
        dist = Sample(
            LeafCTMC(transition_prob_tree, hky_params["frequencies"]),
            sample_shape=hello_alignment.site_count,
        )
        return dist.log_prob(sequences)

    sequences = hello_alignment.get_encoded_sequence_tensor(hello_tensor_tree.taxon_set)

    if function_mode:
        log_prob_fn = tf.function(log_prob_fn)
    res = log_prob_fn(hello_tensor_tree, sequences)
    assert_allclose(res, hello_hky_log_likelihood)


@pytest.mark.parametrize("function_mode", [True, False])
def test_leaf_ctmc_discrete_mixture(
    hello_tensor_tree: TensorflowRootedTree,
    hello_alignment: Alignment,
    hky_params,
    function_mode,
    tensor_constant,
):
    from treeflow.distributions.discretized import DiscretizedDistribution
    from treeflow.distributions.discrete_parameter_mixture import (
        DiscreteParameterMixture,
    )
    from tensorflow_probability.python.distributions import Gamma

    rate_category_count = 5
    rate_distribution = DiscretizedDistribution(
        rate_category_count, Gamma(tensor_constant(2.0), tensor_constant(2.0))
    )

    def log_prob_fn(tree: TensorflowRootedTree, sequences, rate_distribution):

        transition_prob_tree = get_transition_probabilities_tree(
            tree.get_unrooted_tree(),
            HKY(),
            rate_categories=rate_distribution.support,
            **hky_params
        )  # TODO: Better solution for batch dimensions
        site_distributions = DiscreteParameterMixture(
            rate_distribution, LeafCTMC(transition_prob_tree, hky_params["frequencies"])
        )
        dist = Sample(
            site_distributions,
            sample_shape=hello_alignment.site_count,
        )
        return dist.log_prob(sequences)

    if function_mode:
        log_prob_fn = tf.function(log_prob_fn)

    sequences = hello_alignment.get_encoded_sequence_tensor(hello_tensor_tree.taxon_set)
    res = log_prob_fn(hello_tensor_tree, sequences, rate_distribution)

    def site_prob_fn(tree: TensorflowRootedTree, sequences, rate):
        unrooted_tree = tree.get_unrooted_tree()
        transition_prob_tree = get_transition_probabilities_tree(
            unrooted_tree.with_branch_lengths(unrooted_tree.branch_lengths * rate),
            HKY(),
            **hky_params
        )  # TODO: Better solution for batch dimensions
        dist = LeafCTMC(transition_prob_tree, hky_params["frequencies"])
        return dist.prob(sequences)

    site_probs = np.zeros((hello_alignment.site_count, rate_category_count))
    for site in range(hello_alignment.site_count):
        for category in range(rate_category_count):
            site_probs[site, category] = site_prob_fn(
                hello_tensor_tree, sequences[site], rate_distribution.support[category]
            ).numpy()
    expected = np.sum(
        np.log(np.sum(site_probs * rate_distribution.probabilities.numpy(), axis=-1))
    )

    assert_allclose(res.numpy(), expected)


# ---------------------------------------------------------------------------
# Sampling tests
# ---------------------------------------------------------------------------


@pytest.fixture
def two_taxon_transition_prob_tree():
    """Minimal 2-taxon tree: leaves 0,1; root 2; branches 0 and 1."""
    parent_indices = np.array([2, 2], dtype=np.int32)
    child_indices = get_child_indices(parent_indices)
    preorder_indices = get_preorder_indices(child_indices)
    topology = TensorflowTreeTopology(
        parent_indices=tf.constant(parent_indices),
        child_indices=tf.constant(child_indices),
        preorder_indices=tf.constant(preorder_indices),
    )
    state_count = 4
    # Uniform transition probs
    transition_probs = tf.fill([2, state_count, state_count], 1.0 / state_count)
    return TensorflowUnrootedTree(branch_lengths=transition_probs, topology=topology)


def test_LeafCTMC_sample_shape(two_taxon_transition_prob_tree):
    state_count = two_taxon_transition_prob_tree.branch_lengths.shape[-1]
    taxon_count = two_taxon_transition_prob_tree.taxon_count
    frequencies = tf.fill([state_count], 0.25)
    dist = LeafCTMC(two_taxon_transition_prob_tree, frequencies)

    n = 7
    samples = dist.sample(n, seed=(0, 1))

    assert samples.shape == (n, taxon_count, state_count)
    assert samples.dtype == tf.int32
    # Each row must be a valid one-hot vector
    row_sums = tf.reduce_sum(samples, axis=-1)
    assert tf.reduce_all(row_sums == 1).numpy()
    assert tf.reduce_all((samples == 0) | (samples == 1)).numpy()


def test_LeafCTMC_sample_identity_transition(two_taxon_transition_prob_tree):
    """With identity transition matrices, all leaves must have the same state as root."""
    state_count = two_taxon_transition_prob_tree.branch_lengths.shape[-1]
    topology = two_taxon_transition_prob_tree.topology
    # Identity transition: child always equals parent
    identity_probs = tf.eye(state_count)[tf.newaxis, :, :]  # [1, state, state]
    identity_probs = tf.tile(identity_probs, [2, 1, 1])  # [2, state, state]
    transition_probs_tree = TensorflowUnrootedTree(
        branch_lengths=identity_probs, topology=topology
    )
    frequencies = tf.constant([1.0, 0.0, 0.0, 0.0])  # always sample state 0 at root
    dist = LeafCTMC(transition_probs_tree, frequencies)

    samples = dist.sample(10, seed=(0, 1))  # [10, 2, 4]
    # With freq=[1,0,0,0] and identity transitions, all leaves must be state 0
    assert tf.reduce_all(samples[..., 0] == 1).numpy()  # one-hot position 0 always 1


def test_LeafCTMC_sample_seed_reproducibility(two_taxon_transition_prob_tree):
    state_count = two_taxon_transition_prob_tree.branch_lengths.shape[-1]
    frequencies = tf.fill([state_count], 0.25)
    dist = LeafCTMC(two_taxon_transition_prob_tree, frequencies)

    s1 = dist.sample(5, seed=(42, 0))
    s2 = dist.sample(5, seed=(42, 0))
    assert tf.reduce_all(s1 == s2).numpy()


def test_LeafCTMC_sample_prob_positive(two_taxon_transition_prob_tree):
    """Samples must lie in the support: prob should be positive and finite."""
    state_count = two_taxon_transition_prob_tree.branch_lengths.shape[-1]
    frequencies = tf.fill([state_count], 0.25)
    dist = LeafCTMC(two_taxon_transition_prob_tree, frequencies)

    sample = dist.sample(seed=(0, 1))  # [2, 4] int32 one-hot
    # _prob expects float one-hot; cast to match what the likelihood function uses
    p = dist.prob(tf.cast(sample, tf.float32))
    assert tf.math.is_finite(p).numpy()
    assert p.numpy() > 0.0


@pytest.mark.parametrize("function_mode", [True, False])
def test_LeafCTMC_sample_tf_function(two_taxon_transition_prob_tree, function_mode):
    state_count = two_taxon_transition_prob_tree.branch_lengths.shape[-1]
    frequencies = tf.fill([state_count], 0.25)
    dist = LeafCTMC(two_taxon_transition_prob_tree, frequencies)

    def sample_fn():
        return dist.sample(3, seed=(0, 1))

    if function_mode:
        sample_fn = tf.function(sample_fn)

    result = sample_fn()
    assert result.shape == (3, two_taxon_transition_prob_tree.taxon_count, state_count)
