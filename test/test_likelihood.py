import pytest
import numdifftools
from numpy.testing import assert_allclose
import numpy as np
import tensorflow as tf
import treeflow.substitution_model
import treeflow.tensorflow_likelihood
import treeflow.sequences
import treeflow.tree_processing


def test_hky_1cat_likelihood_beast(
    prep_likelihood,
    single_hky_params,
    single_rates,
    single_weights,
    hello_newick_file,
    hello_fasta_file,
):
    subst_model = treeflow.substitution_model.HKY()
    tf_likelihood = prep_likelihood(
        hello_newick_file,
        hello_fasta_file,
        subst_model,
        single_rates,
        single_weights,
        **single_hky_params
    )[0]
    assert_allclose(
        tf_likelihood.compute_likelihood_from_partials(
            single_hky_params["frequencies"], single_weights
        ).numpy(),
        -88.86355638556158,
        atol=1e-4,
    )


def test_hky_1cat_beast_kappa_gradient(
    prep_likelihood,
    single_hky_params,
    single_rates,
    single_weights,
    hello_newick_file,
    hello_fasta_file,
):
    subst_model = treeflow.substitution_model.HKY()
    tf_likelihood, branch_lengths, eigendecomp = prep_likelihood(
        hello_newick_file,
        hello_fasta_file,
        subst_model,
        single_rates,
        single_weights,
        **single_hky_params
    )
    q_diff = subst_model.q_norm_param_differentials(**single_hky_params)["kappa"]
    transition_prob_differential = (
        treeflow.substitution_model.transition_probs_differential(
            q_diff, eigendecomp, branch_lengths, single_rates
        )
    )
    assert_allclose(
        tf_likelihood.compute_derivative(
            transition_prob_differential, single_weights
        ).numpy(),
        0.12373298571565322,
        atol=1e-4,
    )


@pytest.mark.skip(reason="BEAST value is wrong")
def test_hky_1cat_freqA_gradient_beast(
    prep_likelihood,
    single_hky_params,
    single_rates,
    single_weights,
    hello_newick_file,
    hello_fasta_file,
):
    freq_index = 0
    subst_model = treeflow.substitution_model.HKY()
    tf_likelihood, branch_lengths, eigendecomp = prep_likelihood(
        hello_newick_file,
        hello_fasta_file,
        subst_model,
        single_rates,
        single_weights,
        **single_hky_params
    )
    q_diff = subst_model.q_norm_frequency_differentials(**single_hky_params)[freq_index]
    transition_prob_differential = (
        treeflow.substitution_model.transition_probs_differential(
            q_diff, eigendecomp, branch_lengths, single_rates
        )
    )
    # grad = tf_likelihood.compute_frequency_derivative(transition_prob_differential, freq_index, single_weights)
    grad = tf_likelihood.compute_derivative(
        transition_prob_differential, single_weights
    )

    assert_allclose(
        grad.numpy(), 12.658386297868352
    )  # TODO: Correct once BEAST is corrected


def test_hky_1cat_freqA_gradient_num(
    prep_likelihood,
    single_hky_params,
    single_rates,
    single_weights,
    hello_newick_file,
    hello_fasta_file,
):
    freq_index = 0
    subst_model = treeflow.substitution_model.HKY()
    tf_likelihood, branch_lengths, eigendecomp = prep_likelihood(
        hello_newick_file,
        hello_fasta_file,
        subst_model,
        single_rates,
        single_weights,
        **single_hky_params
    )
    q_diff = subst_model.q_norm_frequency_differentials(**single_hky_params)[freq_index]
    transition_prob_differential = (
        treeflow.substitution_model.transition_probs_differential(
            q_diff, eigendecomp, branch_lengths, single_rates
        )
    )
    grad = tf_likelihood.compute_frequency_derivative(
        transition_prob_differential, freq_index, single_weights
    )

    freq_vals = single_hky_params["frequencies"].numpy()
    freqA_val = freq_vals[freq_index]

    def like_func(freqA):
        freq_vals[freq_index] = freqA
        q = subst_model.q_norm(freq_vals, kappa=single_hky_params["kappa"])
        return tf_likelihood.compute_likelihood_expm(
            branch_lengths, single_rates, single_weights, freq_vals, q
        ).numpy()

    num_grad = numdifftools.Derivative(like_func)(freqA_val)

    assert_allclose(grad.numpy(), num_grad, atol=1e-4)


def test_hky_kappa_gradient_tf(
    prep_likelihood, hky_params, weights_rates, hello_newick_file, hello_fasta_file
):
    category_weights, category_rates = weights_rates
    subst_model = treeflow.substitution_model.HKY()
    tf_likelihood, branch_lengths, eigendecomp = prep_likelihood(
        hello_newick_file,
        hello_fasta_file,
        subst_model,
        category_rates,
        category_weights,
        **hky_params
    )
    with tf.GradientTape() as t:
        t.watch(hky_params["kappa"])
        q = subst_model.q_norm(**hky_params)
        likelihood = tf_likelihood.compute_likelihood_expm(
            branch_lengths,
            category_rates,
            category_weights,
            hky_params["frequencies"],
            q,
        )

    tf_grad = t.gradient(likelihood, hky_params["kappa"])

    q_diff = subst_model.q_norm_param_differentials(**hky_params)["kappa"]
    transition_prob_differential = (
        treeflow.substitution_model.transition_probs_differential(
            q_diff, eigendecomp, branch_lengths, category_rates
        )
    )
    grad = tf_likelihood.compute_derivative(
        transition_prob_differential, category_weights
    )
    assert_allclose(grad.numpy(), tf_grad.numpy(), atol=1e-4)


def test_hky_freq_gradient_tf(
    prep_likelihood,
    hky_params,
    weights_rates,
    hello_newick_file,
    hello_fasta_file,
    freq_index,
):
    category_weights, category_rates = weights_rates
    subst_model = treeflow.substitution_model.HKY()
    tf_likelihood, branch_lengths, eigendecomp = prep_likelihood(
        hello_newick_file,
        hello_fasta_file,
        subst_model,
        category_rates,
        category_weights,
        **hky_params
    )
    with tf.GradientTape() as t:
        t.watch(hky_params["frequencies"])
        q = subst_model.q_norm(**hky_params)
        likelihood = tf_likelihood.compute_likelihood_expm(
            branch_lengths,
            category_rates,
            category_weights,
            hky_params["frequencies"],
            q,
        )

    tf_grad = t.gradient(likelihood, hky_params["frequencies"])[freq_index]

    q_diff = subst_model.q_norm_frequency_differentials(**hky_params)[freq_index]
    transition_prob_differential = (
        treeflow.substitution_model.transition_probs_differential(
            q_diff, eigendecomp, branch_lengths, category_rates
        )
    )
    grad = tf_likelihood.compute_frequency_derivative(
        transition_prob_differential, freq_index, category_weights
    )
    assert_allclose(grad.numpy(), tf_grad.numpy(), atol=1e-4)


def test_hky_branch_gradient_tf(
    prep_likelihood, hky_params, weights_rates, hello_newick_file, hello_fasta_file
):
    category_weights, category_rates = weights_rates
    subst_model = treeflow.substitution_model.HKY()
    tf_likelihood, branch_lengths, eigendecomp = prep_likelihood(
        hello_newick_file,
        hello_fasta_file,
        subst_model,
        category_rates,
        category_weights,
        **hky_params
    )
    with tf.GradientTape() as t:
        t.watch(branch_lengths)
        likelihood = tf_likelihood.compute_likelihood(
            branch_lengths,
            category_rates,
            category_weights,
            hky_params["frequencies"],
            eigendecomp,
        )
    q = subst_model.q_norm(**hky_params)
    tf_grad = t.gradient(likelihood, branch_lengths)
    grad = tf_likelihood.compute_branch_length_derivatives(
        q, category_rates, category_weights
    )
    assert_allclose(grad.numpy(), tf_grad.numpy(), atol=1e-4)


def test_hky_rate_gradient_tf(
    prep_likelihood, hky_params, weights_rates, hello_newick_file, hello_fasta_file
):
    category_weights, category_rates = weights_rates
    subst_model = treeflow.substitution_model.HKY()
    tf_likelihood, branch_lengths, eigendecomp = prep_likelihood(
        hello_newick_file,
        hello_fasta_file,
        subst_model,
        category_rates,
        category_weights,
        **hky_params
    )
    with tf.GradientTape() as t:
        t.watch(category_rates)
        likelihood = tf_likelihood.compute_likelihood(
            branch_lengths,
            category_rates,
            category_weights,
            hky_params["frequencies"],
            eigendecomp,
        )
    q = subst_model.q_norm(**hky_params)
    tf_grad = t.gradient(likelihood, category_rates)
    grad = tf_likelihood.compute_rate_derivatives(q, branch_lengths, category_weights)
    assert_allclose(grad.numpy(), tf_grad.numpy(), atol=1e-4)


def test_hky_weight_gradient_tf(
    prep_likelihood, hky_params, weights_rates, hello_newick_file, hello_fasta_file
):
    category_weights, category_rates = weights_rates
    subst_model = treeflow.substitution_model.HKY()
    tf_likelihood, branch_lengths, eigendecomp = prep_likelihood(
        hello_newick_file,
        hello_fasta_file,
        subst_model,
        category_rates,
        category_weights,
        **hky_params
    )
    with tf.GradientTape() as t:
        t.watch(category_weights)
        likelihood = tf_likelihood.compute_likelihood(
            branch_lengths,
            category_rates,
            category_weights,
            hky_params["frequencies"],
            eigendecomp,
        )
    tf_grad = t.gradient(likelihood, category_weights)
    grad = tf_likelihood.compute_weight_derivatives(category_weights)
    assert_allclose(grad.numpy(), tf_grad.numpy(), atol=1e-4)
