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

    rate_category_count = 4
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
