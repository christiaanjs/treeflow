import pytest
from numpy.testing import assert_allclose
import numpy as np
import tensorflow as tf
import treeflow.substitution_model
import treeflow.tensorflow_likelihood

def prep_likelihood(newick_file, fasta_file, subst_model, rates, weights, frequencies, **subst_params):
    eigendecomp = subst_model.eigen(frequencies, **subst_params)
    tf_likelihood = treeflow.tensorflow_likelihood.TensorflowLikelihood(newick_file=newick_file, fasta_file=fasta_file, category_count=len(rates))
    branch_lengths = tf.convert_to_tensor(tf_likelihood.get_init_branch_lengths())
    transition_probs = treeflow.substitution_model.transition_probs(eigendecomp, rates, branch_lengths)
    tf_likelihood.compute_postorder_partials(transition_probs)
    tf_likelihood.init_preorder_partials(frequencies)
    tf_likelihood.compute_preorder_partials(transition_probs)
    return tf_likelihood, branch_lengths, eigendecomp

@pytest.fixture
def single_rates():
    return tf.convert_to_tensor(np.array([1.0]))

@pytest.fixture
def single_weights():
    return tf.convert_to_tensor(np.array([1.0]))

def test_hky_1cat_likelihood(single_hky_params, single_rates, single_weights, hello_newick_file, hello_fasta_file):
    subst_model = treeflow.substitution_model.HKY()
    tf_likelihood = prep_likelihood(hello_newick_file, hello_fasta_file, subst_model, single_rates, single_weights, **single_hky_params)[0]
    assert_allclose(tf_likelihood.compute_likelihood_from_partials(single_hky_params['frequencies'], single_weights).numpy(), -88.86355638556158)

def test_hky_1cat_beast_kappa_gradient(single_hky_params, single_rates, single_weights, hello_newick_file, hello_fasta_file):
    subst_model = treeflow.substitution_model.HKY()
    tf_likelihood, branch_lengths, eigendecomp = prep_likelihood(hello_newick_file, hello_fasta_file, subst_model, single_rates, single_weights, **single_hky_params)
    q_diff = subst_model.q_norm_param_differentials(**single_hky_params)['kappa']
    transition_prob_differential = treeflow.substitution_model.transition_probs_differential(q_diff, eigendecomp, branch_lengths, single_rates)
    assert_allclose(tf_likelihood.compute_derivative(transition_prob_differential, single_weights).numpy(), 0.12373298571565322)

def test_hky_1cat_tf_branch_gradient(single_hky_params, single_rates, single_weights, hello_newick_file, hello_fasta_file):
    subst_model = treeflow.substitution_model.HKY()
    tf_likelihood, branch_lengths, eigendecomp = prep_likelihood(hello_newick_file, hello_fasta_file, subst_model, single_rates, single_weights, **single_hky_params)
    with tf.GradientTape() as t:
        t.watch(branch_lengths)
        likelihood = tf_likelihood.compute_likelihood(branch_lengths, single_rates, single_weights, single_hky_params['frequencies'], eigendecomp)
    q = subst_model.q_norm(**single_hky_params)
    tf_grad = t.gradient(likelihood, branch_lengths)
    grad = tf_likelihood.compute_branch_length_derivatives(q, single_weights)
    assert_allclose(grad.numpy(), tf_grad.numpy())