import pytest
from numpy.testing import assert_allclose
import treeflow.substitution_model
import treeflow.sequences
import treeflow.tree_processing
import tensorflow as tf

@pytest.mark.parametrize('grad_param', ['kappa', 'frequencies', 'rates', 'weights', 'branch_lengths'])
def test_log_prob_custom_gradient_hky(prep_likelihood, hky_params, weights_rates, hello_newick_file, hello_fasta_file, grad_param):
    subst_model = treeflow.substitution_model.HKY()
    category_weights, category_rates = weights_rates
    tf_likelihood, branch_lengths, eigen = prep_likelihood(hello_newick_file, hello_fasta_file, subst_model, category_rates, category_weights, **hky_params)
    
    tree, taxon_names = treeflow.tree_processing.parse_newick(hello_newick_file)

    value = treeflow.sequences.get_encoded_sequences(hello_fasta_file, taxon_names) # TODO: return from prep_likelihood
    log_prob = treeflow.sequences.log_prob_conditioned(value, tree['topology'], len(category_rates))

    power = 2.0

    if grad_param == 'kappa':
        var = hky_params['kappa']
        q_diff = subst_model.q_norm_param_differentials(**hky_params)['kappa']
        transition_probs_differential = treeflow.substitution_model.transition_probs_differential(q_diff, eigen, branch_lengths, category_rates)
        expected_ll_grad = tf_likelihood.compute_derivative(transition_probs_differential, category_weights)
    elif grad_param == 'frequencies':
        var = hky_params['frequencies']
        q_diffs = subst_model.q_norm_frequency_differentials(**hky_params)
        transition_probs_differentials = [treeflow.substitution_model.transition_probs_differential(q_diff, eigen, branch_lengths, category_rates) for q_diff in q_diffs]
        expected_ll_grad = tf.stack([tf_likelihood.compute_frequency_derivative(transition_probs_differential, i, category_weights) for i, transition_probs_differential in enumerate(transition_probs_differentials)])
    elif grad_param == 'rates':
        var = category_rates
        q = subst_model.q_norm(**hky_params)
        expected_ll_grad = tf_likelihood.compute_rate_derivatives(q, branch_lengths, category_weights)
    elif grad_param == 'weights':
        var = category_weights
        expected_ll_grad = tf_likelihood.compute_weight_derivatives(category_weights)
    elif grad_param == 'branch_lengths':
        var = branch_lengths
        q = subst_model.q_norm(**hky_params)
        expected_ll_grad = tf_likelihood.compute_branch_length_derivatives(q, category_rates, category_weights) 

    with tf.GradientTape() as t:
        t.watch(var)
        ll = log_prob(branch_lengths, subst_model, category_weights, category_rates, **hky_params)
        y = ll ** power
    res = t.gradient(y, var)

    expected = power * ll * expected_ll_grad  
    assert_allclose(res.numpy(), expected.numpy())
    
