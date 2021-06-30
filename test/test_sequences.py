from re import M

from numpy.lib.function_base import _parse_gufunc_signature
from treeflow import tensorflow_likelihood
import pytest
from numpy.testing import assert_allclose
import treeflow.substitution_model
import treeflow.sequences
import treeflow.tree_processing
import tensorflow as tf


def test_log_prob_optional_custom_gradient(
    hky_params, weights_rates, hello_newick_file, hello_fasta_file
):
    subst_model = treeflow.substitution_model.HKY()
    category_weights, category_rates = weights_rates

    tree, taxon_names = treeflow.tree_processing.parse_newick(hello_newick_file)

    value = treeflow.sequences.get_encoded_sequences(hello_fasta_file, taxon_names)
    log_prob_autodiff = treeflow.sequences.log_prob_conditioned(
        value, tree["topology"], len(category_rates)
    )


@pytest.mark.parametrize("custom_gradient", [True, False])
@pytest.mark.parametrize("function_mode", [True, False])
@pytest.mark.parametrize(
    "grad_param", ["kappa", "frequencies", "rates", "weights", "branch_lengths"]
)
def test_log_prob_custom_gradient_hky(
    prep_likelihood,
    hky_params,
    weights_rates,
    hello_newick_file,
    hello_fasta_file,
    custom_gradient,
    function_mode,
    grad_param,
):
    subst_model = treeflow.substitution_model.HKY()
    category_weights, category_rates = weights_rates
    tf_likelihood, branch_lengths, eigen = prep_likelihood(
        hello_newick_file,
        hello_fasta_file,
        subst_model,
        category_rates,
        category_weights,
        **hky_params
    )

    tree, taxon_names = treeflow.tree_processing.parse_newick(hello_newick_file)

    value = treeflow.sequences.get_encoded_sequences(
        hello_fasta_file, taxon_names
    )  # TODO: return from prep_likelihood
    log_prob = treeflow.sequences.log_prob_conditioned(
        value, tree["topology"], len(category_rates), custom_gradient=custom_gradient
    )

    power = 2.0
    if grad_param == "kappa":
        var = hky_params["kappa"]
        q_diff = subst_model.q_norm_param_differentials(**hky_params)["kappa"]
        transition_probs_differential = (
            treeflow.substitution_model.transition_probs_differential(
                q_diff, eigen, branch_lengths, category_rates
            )
        )
        expected_ll_grad = tf_likelihood.compute_derivative(
            transition_probs_differential, category_weights
        )
    elif grad_param == "frequencies":
        var = hky_params["frequencies"]
        q_diffs = subst_model.q_norm_frequency_differentials(**hky_params)
        transition_probs_differentials = [
            treeflow.substitution_model.transition_probs_differential(
                q_diff, eigen, branch_lengths, category_rates
            )
            for q_diff in q_diffs
        ]
        expected_ll_grad = tf.stack(
            [
                tf_likelihood.compute_frequency_derivative(
                    transition_probs_differential, i, category_weights
                )
                for i, transition_probs_differential in enumerate(
                    transition_probs_differentials
                )
            ]
        )
    elif grad_param == "rates":
        var = category_rates
        q = subst_model.q_norm(**hky_params)
        expected_ll_grad = tf_likelihood.compute_rate_derivatives(
            q, branch_lengths, category_weights
        )
    elif grad_param == "weights":
        var = category_weights
        expected_ll_grad = tf_likelihood.compute_weight_derivatives(category_weights)
    elif grad_param == "branch_lengths":
        var = branch_lengths
        q = subst_model.q_norm(**hky_params)
        expected_ll_grad = tf_likelihood.compute_branch_length_derivatives(
            q, category_rates, category_weights
        )

    def grad(x):
        with tf.GradientTape() as t:
            t.watch(x)
            local_branch_lengths = branch_lengths
            local_category_weights = category_weights
            local_category_rates = category_rates
            local_hky_params = hky_params
            if grad_param == "kappa":
                local_hky_params = dict(kappa=x, frequencies=hky_params["frequencies"])
            elif grad_param == "frequencies":
                local_hky_params = dict(kappa=hky_params["kappa"], frequencies=x)
            elif grad_param == "rates":
                local_category_rates = x
            elif grad_param == "weights":
                local_category_weights = x
            elif grad_param == "branch_lengths":
                local_branch_lengths = x
            ll = log_prob(
                local_branch_lengths,
                subst_model,
                local_category_weights,
                local_category_rates,
                **local_hky_params
            )
            y = ll ** power
            return t.gradient(y, x)

    if function_mode:
        log_prob = tf.function(log_prob)
        grad = tf.function(grad)

    ll = log_prob(
        branch_lengths, subst_model, category_weights, category_rates, **hky_params
    )
    res = grad(var)
    expected = power * ll * expected_ll_grad
    assert_allclose(res.numpy(), expected.numpy())


branch_lengths = [[0.3, 0.4, 1.2, 0.7], [0.9, 0.2, 2.3, 1.4]]
heights = [[0.2, 0.1, 0.0, 0.5, 1.2], [0.0, 0.7, 0.0, 0.9, 2.3]]
parent_indices = [3, 3, 4, 4]


@pytest.mark.parametrize("custom_gradient", [True, False])
@pytest.mark.parametrize("function_mode", [True, False])
def test_log_prob_branch_only_hky(
    hky_params,
    weights_rates,
    hello_newick_file,
    hello_fasta_file,
    custom_gradient,
    function_mode,
):
    subst_model = treeflow.substitution_model.HKY()
    category_weights, category_rates = weights_rates
    tree, taxon_names = treeflow.tree_processing.parse_newick(hello_newick_file)

    value = treeflow.sequences.get_encoded_sequences(hello_fasta_file, taxon_names)
    (
        log_prob,
        tensorflow_likelihood,
    ) = treeflow.sequences.log_prob_conditioned_branch_only(
        value,
        tree["topology"],
        len(category_weights),
        subst_model,
        category_weights,
        category_rates,
        hky_params["frequencies"],
        kappa=hky_params["kappa"],
        custom_gradient=custom_gradient,
    )

    def grad(branch_lengths):
        with tf.GradientTape() as t:
            t.watch(branch_lengths)
            val = log_prob(branch_lengths)
            return t.gradient(val, branch_lengths)

    if function_mode:
        log_prob = tf.function(log_prob)
        grad = tf.function(grad)

    branch_lengths = treeflow.sequences.get_branch_lengths(tree)
    log_prob(branch_lengths)
    grad(branch_lengths)


@pytest.mark.parametrize(
    "branch_lengths,heights,parent_indices",
    [
        (branch_lengths[0], heights[0], parent_indices),
        (branch_lengths[1], heights[1], parent_indices),
        (branch_lengths, heights, [parent_indices, parent_indices]),
    ],
)
def test_get_branch_lengths(branch_lengths, heights, parent_indices, function_mode):
    test_function = (
        tf.function(treeflow.sequences.get_branch_lengths)
        if function_mode
        else treeflow.sequences.get_branch_lengths
    )
    tree = treeflow.tree_processing.tree_to_tensor(
        dict(heights=heights, topology=dict(parent_indices=parent_indices))
    )
    assert_allclose(test_function(tree), branch_lengths)
