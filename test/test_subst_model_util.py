import pytest
from treeflow.substitution_model import JC, HKY, normalising_constant, normalised_differential, transition_probs, transition_probs_differential
import numpy as np
from numpy.testing import assert_allclose
import tensorflow as tf

def get_transition_probs_vals(subst_model, branch_lengths, category_rates, **params):
    branch_lengths_ = tf.convert_to_tensor(branch_lengths)
    rates_ = tf.convert_to_tensor(category_rates)
    eigen = subst_model.eigen(**params)
    return transition_probs(eigen, rates_, branch_lengths_).numpy()

def test_transition_probs_hky_rowsum(hky_params, branch_lengths, category_rates):
    transition_prob_vals = get_transition_probs_vals(JC(), branch_lengths, category_rates, **hky_params)
    row_sums = np.sum(transition_prob_vals, axis=3)
    assert_allclose(1.0, row_sums)

def test_transition_probs_jc(branch_lengths, category_rates):
    transition_prob_vals = get_transition_probs_vals(JC(), branch_lengths, category_rates)

    diag_mask = np.eye(4, dtype=bool)
    diag_i, diag_j = np.where(diag_mask)
    non_diag_i, non_diag_j = np.where(~diag_mask)

    for rate_index, rate in enumerate(category_rates):
        for branch_index, branch_length in enumerate(branch_lengths):
            d = rate * branch_length
            print(d)
            probs = transition_prob_vals[branch_index, rate_index]
            print(probs)
            assert_allclose(0.25 + 0.75*np.exp(-4*d/3), probs[diag_i, diag_j])
            assert_allclose(0.25 - 0.25*np.exp(-4*d/3), probs[non_diag_i, non_diag_j])


@pytest.mark.parametrize('param_key', HKY().param_keys() + ['frequencies'])
def test_normalisation_gradient(hky_params, param_key):
    subst_model = HKY()
    params = hky_params
    with tf.GradientTape() as t:
        t.watch(params[param_key])
        q_norm = subst_model.q_norm(**params)
    q = subst_model.q(**params)
    norm_const = normalising_constant(q, params['frequencies'])
    tf_jac = t.jacobian(q_norm, params[param_key])
    if param_key != 'frequencies':
        norm_diffs = subst_model.q_norm_param_differentials(**params)[param_key].numpy()
        tf_diffs = tf_jac.numpy()
    else:
        norm_diffs = subst_model.q_norm_frequency_differentials(**params)
        tf_diffs = tf.transpose(tf_jac, perm=[2, 0, 1]).numpy()
    assert_allclose(tf_diffs, norm_diffs)

@pytest.mark.parametrize('param_key', HKY().param_keys() + ['frequencies'])
@pytest.mark.parametrize('inv_mult', ['True', 'False'])
def test_transition_prob_differential(hky_params, param_key, inv_mult):
    branch_lengths = tf.convert_to_tensor(np.array([0.1]))#, 0.9, 2.2]))
    subst_model = HKY()
    params = hky_params
    with tf.GradientTape() as t:
        t.watch(params[param_key])
        eigen = subst_model.eigen(**params)
        transition_probs_vals = transition_probs(eigen, branch_lengths)
    tf_jacs = t.jacobian(transition_probs_vals, params[param_key]) 
    transition_probs_inv = tf.linalg.inv(transition_probs_vals)

    if param_key != 'frequencies':
        q_diffs = subst_model.q_norm_param_differentials(**params)[param_key]
        diffs = transition_probs_differential(q_diffs, eigen,  branch_lengths, inv_mult=inv_mult)
        tf_diffs = (transition_probs_inv @ tf_jacs) if inv_mult else tf_jacs
    else:
        freq_index = 2
        q_diffs = subst_model.q_norm_frequency_differentials(**params)
        diffs = transition_probs_differential(q_diffs[freq_index], eigen, branch_lengths, inv_mult=inv_mult)
        tf_diffs = (transition_probs_inv @ tf_jacs[:, :, :, freq_index]) if inv_mult else tf_jacs[:, :, :, freq_index]
    assert_allclose(tf_diffs.numpy(), diffs.numpy())

