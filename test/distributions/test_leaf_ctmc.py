import pytest
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.tree.rooted.tensorflow_rooted_tree import (
    TensorflowRootedTree,
    convert_tree_to_tensor,
)
from treeflow.tree.topology.tensorflow_tree_topology import numpy_topology_to_tensor
import tensorflow as tf
from treeflow.distributions.leaf_ctmc import LeafCTMC
from treeflow.tree.unrooted.tensorflow_unrooted_tree import TensorflowUnrootedTree


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


def test_leaf_ctmc_event_shape(transition_prob_tree):
    state_count = transition_prob_tree.branch_lengths.numpy().shape[-1]
    dist = LeafCTMC(transition_prob_tree)
    assert dist.event_shape == (transition_prob_tree.taxon_count, state_count)


def test_leaf_ctmc_event_shape_tensor(transition_prob_tree):
    state_count = transition_prob_tree.branch_lengths.numpy().shape[-1]
    dist = LeafCTMC(transition_prob_tree)
    assert tuple(dist.event_shape_tensor().numpy()) == (
        transition_prob_tree.taxon_count,
        state_count,
    )
