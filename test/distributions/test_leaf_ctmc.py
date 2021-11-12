import pytest
import tensorflow as tf
from tensorflow_probability.python.distributions import Sample
from numpy.testing import assert_allclose
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.tree.unrooted.tensorflow_unrooted_tree import TensorflowUnrootedTree
from treeflow.tree.rooted.tensorflow_rooted_tree import (
    TensorflowRootedTree,
    convert_tree_to_tensor,
)
from treeflow.tree.topology.tensorflow_tree_topology import numpy_topology_to_tensor
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
            flat_tree_test_data.heights,
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


def test_leaf_ctmc_event_shape(transition_prob_tree, hky_params):
    state_count = transition_prob_tree.branch_lengths.numpy().shape[-1]
    dist = LeafCTMC(transition_prob_tree, hky_params["frequencies"])
    assert dist.event_shape == (transition_prob_tree.taxon_count, state_count)


def test_leaf_ctmc_event_shape_tensor(transition_prob_tree, hky_params):
    state_count = transition_prob_tree.branch_lengths.numpy().shape[-1]
    dist = LeafCTMC(transition_prob_tree, hky_params["frequencies"])
    assert tuple(dist.event_shape_tensor().numpy()) == (
        transition_prob_tree.taxon_count,
        state_count,
    )


def test_leaf_ctmc_log_prob_over_sites(
    hello_tensor_tree: TensorflowRootedTree,
    hello_alignment: Alignment,
    hky_params,
    hello_hky_log_likelihood,
):
    """Integration-style test"""
    transition_prob_tree = get_transition_probabilities_tree(
        hello_tensor_tree.get_unrooted_tree(), HKY(), **hky_params
    )  # TODO: Better solution for batch dimensions
    dist = Sample(
        LeafCTMC(transition_prob_tree, hky_params["frequencies"]),
        sample_shape=hello_alignment.site_count,
    )
    sequences = hello_alignment.get_encoded_sequence_tensor(hello_tensor_tree.taxon_set)
    res = dist.log_prob(sequences)
    assert_allclose(res, hello_hky_log_likelihood)
