"""
Tests for RescaledLeafCTMC.

Verifies that RescaledLeafCTMC.log_prob produces results that match
LeafCTMC.log_prob on the same inputs, and that it gives the correct
log-likelihood on the BEAST benchmark data set.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
import tensorflow as tf
from tensorflow_probability.python.distributions import Sample

from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.tree.unrooted.tensorflow_unrooted_tree import TensorflowUnrootedTree
from treeflow.tree.rooted.tensorflow_rooted_tree import (
    TensorflowRootedTree,
    convert_tree_to_tensor,
)
from treeflow.tree.topology.tensorflow_tree_topology import numpy_topology_to_tensor
from treeflow.evolution.substitution.nucleotide.hky import HKY
from treeflow.distributions.leaf_ctmc import LeafCTMC
from treeflow.distributions.leaf_ctmc_rescaled import RescaledLeafCTMC
from treeflow.evolution.seqio import Alignment
from treeflow.evolution.substitution.probabilities import (
    get_transition_probabilities_tree,
)


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _make_uniform_transition_prob_tree(flat_tree_test_data, state_count=4, dtype=tf.float64):
    tree = convert_tree_to_tensor(
        NumpyRootedTree(
            node_heights=flat_tree_test_data.node_heights,
            sampling_times=flat_tree_test_data.sampling_times,
            parent_indices=flat_tree_test_data.parent_indices,
        )
    ).get_unrooted_tree()
    transition_probs = tf.fill(
        tree.branch_lengths.shape + (state_count, state_count),
        tf.constant(1.0 / state_count, dtype=dtype),
    )
    return TensorflowUnrootedTree(
        branch_lengths=transition_probs,
        topology=numpy_topology_to_tensor(tree.topology),
    )


# ---------------------------------------------------------------------------
# Single-site: RescaledLeafCTMC.log_prob == LeafCTMC.log_prob
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function_mode", [True, False])
def test_rescaled_matches_leaf_ctmc_single_site(
    flat_tree_test_data, hky_params, function_mode
):
    """log_prob from rescaled version equals log_prob from plain version."""
    state_count = 4
    transition_prob_tree = _make_uniform_transition_prob_tree(
        flat_tree_test_data, state_count
    )
    frequencies = tf.fill([state_count], tf.constant(0.25, dtype=hky_params["frequencies"].dtype))

    dist_plain = LeafCTMC(transition_prob_tree, frequencies)
    dist_rescaled = RescaledLeafCTMC(transition_prob_tree, frequencies)

    # Build a single one-hot observation [leaf, state] — float64 to match dist dtype
    taxon_count = transition_prob_tree.taxon_count
    x = tf.cast(tf.one_hot(tf.zeros(taxon_count, dtype=tf.int32), state_count), frequencies.dtype)

    def compute(x):
        lp_plain = tf.math.log(dist_plain.prob(x))
        lp_rescaled = dist_rescaled.log_prob(x)
        return lp_plain, lp_rescaled

    if function_mode:
        compute = tf.function(compute)

    lp_plain, lp_rescaled = compute(x)
    assert_allclose(lp_rescaled.numpy(), lp_plain.numpy(), rtol=1e-5)


# ---------------------------------------------------------------------------
# HKY benchmark: matches expected BEAST log-likelihood
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function_mode", [True, False])
def test_rescaled_log_prob_hky_beast(
    hello_tensor_tree: TensorflowRootedTree,
    hello_alignment: Alignment,
    hky_params,
    hello_hky_log_likelihood: float,
    function_mode: bool,
):
    """RescaledLeafCTMC gives the same total log-likelihood as the BEAST reference."""

    def log_prob_fn(tree, sequences):
        transition_prob_tree = get_transition_probabilities_tree(
            tree.get_unrooted_tree(), HKY(), **hky_params
        )
        dist = Sample(
            RescaledLeafCTMC(transition_prob_tree, hky_params["frequencies"]),
            sample_shape=hello_alignment.site_count,
        )
        return dist.log_prob(sequences)

    sequences = hello_alignment.get_encoded_sequence_tensor(hello_tensor_tree.taxon_set)

    if function_mode:
        log_prob_fn = tf.function(log_prob_fn)

    res = log_prob_fn(hello_tensor_tree, sequences)
    assert_allclose(res.numpy(), hello_hky_log_likelihood, rtol=1e-5)


# ---------------------------------------------------------------------------
# Multi-site batch: rescaled log_prob == plain log_prob over all sites
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function_mode", [True, False])
def test_rescaled_matches_leaf_ctmc_multi_site(
    hello_tensor_tree: TensorflowRootedTree,
    hello_alignment: Alignment,
    hky_params,
    function_mode: bool,
):
    """Per-site log_prob from both distributions agree across all sites."""

    def compute(tree, sequences):
        transition_prob_tree = get_transition_probabilities_tree(
            tree.get_unrooted_tree(), HKY(), **hky_params
        )
        dist_plain = LeafCTMC(transition_prob_tree, hky_params["frequencies"])
        dist_rescaled = RescaledLeafCTMC(transition_prob_tree, hky_params["frequencies"])
        lp_plain = tf.math.log(dist_plain.prob(sequences))
        lp_rescaled = dist_rescaled.log_prob(sequences)
        return lp_plain, lp_rescaled

    sequences = hello_alignment.get_encoded_sequence_tensor(hello_tensor_tree.taxon_set)

    if function_mode:
        compute = tf.function(compute)

    lp_plain, lp_rescaled = compute(hello_tensor_tree, sequences)
    assert_allclose(lp_rescaled.numpy(), lp_plain.numpy(), rtol=1e-5)


# ---------------------------------------------------------------------------
# Batched parameters (rate categories): rescaled matches plain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function_mode", [True, False])
def test_rescaled_matches_leaf_ctmc_rate_categories(
    hello_tensor_tree: TensorflowRootedTree,
    hello_alignment: Alignment,
    hky_params,
    function_mode: bool,
):
    """Batched rate-category dimension: rescaled log_prob == log(plain prob)."""
    rate_categories = tf.constant([0.5, 1.0, 2.0], dtype=hky_params["frequencies"].dtype)

    def compute(tree, sequences, rate_categories):
        transition_prob_tree = get_transition_probabilities_tree(
            tree.get_unrooted_tree(),
            HKY(),
            rate_categories=rate_categories,
            **hky_params,
        )
        dist_plain = LeafCTMC(transition_prob_tree, hky_params["frequencies"])
        dist_rescaled = RescaledLeafCTMC(
            transition_prob_tree, hky_params["frequencies"]
        )
        # sequences has shape [sites, leaves, states].  The distribution
        # batch shape is (rate_categories,), so we add an axis so the
        # input broadcasts: [sites, 1, leaves, states] → [sites, rate_categories, leaves, states].
        sequences_expanded = sequences[:, tf.newaxis, :, :]
        lp_plain = tf.math.log(dist_plain.prob(sequences_expanded))
        lp_rescaled = dist_rescaled.log_prob(sequences_expanded)
        return lp_plain, lp_rescaled

    sequences = hello_alignment.get_encoded_sequence_tensor(hello_tensor_tree.taxon_set)

    if function_mode:
        compute = tf.function(compute)

    lp_plain, lp_rescaled = compute(hello_tensor_tree, sequences, rate_categories)
    assert_allclose(lp_rescaled.numpy(), lp_plain.numpy(), rtol=1e-5)


# ---------------------------------------------------------------------------
# Event / batch shape properties mirror LeafCTMC
# ---------------------------------------------------------------------------


def test_rescaled_event_shape(flat_tree_test_data, hky_params):
    state_count = 4
    transition_prob_tree = _make_uniform_transition_prob_tree(
        flat_tree_test_data, state_count
    )
    frequencies = tf.fill([state_count], tf.constant(0.25, dtype=hky_params["frequencies"].dtype))
    dist = RescaledLeafCTMC(transition_prob_tree, frequencies)
    assert tuple(dist.event_shape) == (transition_prob_tree.taxon_count, state_count)


def test_rescaled_batch_shape_with_rate_categories(
    tree_test_data, flat_tree_test_data
):
    tree = convert_tree_to_tensor(
        NumpyRootedTree(
            node_heights=tree_test_data.node_heights,
            sampling_times=tree_test_data.sampling_times,
            parent_indices=flat_tree_test_data.parent_indices,
        )
    ).get_unrooted_tree()
    state_count = 4
    rate_category_count = 3
    transition_probs = tf.fill(
        tree.branch_lengths.shape[:-1]
        + (rate_category_count,)
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
        batch_shape + (rate_category_count, state_count),
        tf.constant(0.25, dtype=transition_probs.dtype),
    )
    dist = RescaledLeafCTMC(transition_probs_tree, frequencies)
    assert dist.batch_shape == batch_shape + (rate_category_count,)
