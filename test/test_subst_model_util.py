import pytest
from treeflow.substitution_model import JC, HKY, normalising_constant, normalised_differential, transition_probs, transition_probs_expm, transition_probs_differential
import numpy as np
from numpy.testing import assert_allclose
import tensorflow as tf

def get_transition_probs_vals(subst_model, branch_lengths, category_rates, **params):
    branch_lengths_ = tf.convert_to_tensor(branch_lengths, dtype=tf.float32)
    rates_ = tf.convert_to_tensor(category_rates, dtype=tf.float32)
    eigen = subst_model.eigen(**params)
    return transition_probs(eigen, rates_, branch_lengths_).numpy()

def test_transition_probs_hky_rowsum(hky_params, branch_lengths, category_rates):
    transition_prob_vals = get_transition_probs_vals(JC(), branch_lengths, category_rates, **hky_params)
    row_sums = np.sum(transition_prob_vals, axis=3)
    assert_allclose(1.0, row_sums, rtol=1e-6)

def test_transition_probs_jc(branch_lengths, category_rates):
    transition_prob_vals = get_transition_probs_vals(JC(), branch_lengths, category_rates)

    diag_mask = np.eye(4, dtype=bool)
    diag_i, diag_j = np.where(diag_mask)
    non_diag_i, non_diag_j = np.where(~diag_mask)

    for rate_index, rate in enumerate(category_rates):
        for branch_index, branch_length in enumerate(branch_lengths):
            d = rate * branch_length
            probs = transition_prob_vals[branch_index, rate_index]
            assert_allclose(0.25 + 0.75*np.exp(-4*d/3), probs[diag_i, diag_j], rtol=1e-6)
            assert_allclose(0.25 - 0.25*np.exp(-4*d/3), probs[non_diag_i, non_diag_j], atol=1e-7)

def test_transition_probs_hky_expm(hky_params, branch_lengths, category_rates):
    subst_model = HKY()
    eigen = subst_model.eigen(**hky_params)
    probs_eigen = transition_probs(eigen, category_rates, branch_lengths)
    probs_expm = transition_probs_expm(subst_model.q_norm(**hky_params), category_rates, branch_lengths)
    assert_allclose(probs_eigen.numpy(), probs_expm.numpy(), atol=1e-6)

@pytest.mark.parametrize('param_key', HKY().param_keys() + ['frequencies'])
def test_normalisation_gradient(hky_params, param_key):
    subst_model = HKY()
    params = hky_params
    with tf.GradientTape() as t:
        t.watch(params[param_key])
        q_norm = subst_model.q_norm(**params)
    tf_jac = t.jacobian(q_norm, params[param_key])
    if param_key != 'frequencies':
        norm_diffs = subst_model.q_norm_param_differentials(**params)[param_key].numpy()
        tf_diffs = tf_jac.numpy()
    else:
        norm_diffs = subst_model.q_norm_frequency_differentials(**params)
        tf_diffs = tf.transpose(tf_jac, perm=[2, 0, 1]).numpy()

    assert_allclose(tf_diffs, norm_diffs)

@pytest.mark.parametrize('inv_mult', [False, True])
def test_transition_prob_differential_tf_hky_kappa(hky_params, category_rates, inv_mult):
    param_key = 'kappa'
    branch_lengths = tf.convert_to_tensor(np.array([0.9, 2.2]), dtype=tf.float32)
    subst_model = HKY()
    params = hky_params
    with tf.GradientTape(persistent=True) as t:
        t.watch(params[param_key])
        q_norm = subst_model.q_norm(**params)
        transition_probs_vals = transition_probs_expm(q_norm, category_rates, branch_lengths)
    tf_jacs = t.jacobian(transition_probs_vals, params[param_key], experimental_use_pfor=False)
    transition_probs_inv = tf.linalg.inv(transition_probs_vals)

    eigen = subst_model.eigen(**params)

    q_diffs = subst_model.q_norm_param_differentials(**params)[param_key]
    diffs = transition_probs_differential(q_diffs, eigen, branch_lengths, category_rates, inv_mult=inv_mult)
    tf_diffs = (transition_probs_inv @ tf_jacs) if inv_mult else tf_jacs

    assert_allclose(tf_diffs.numpy(), diffs.numpy())

@pytest.mark.parametrize('inv_mult', [False, True])
def test_transition_prob_differential_tf_hky_frequencies(hky_params, category_rates, inv_mult, freq_index):
    branch_lengths = tf.convert_to_tensor(np.array([0.9, 2.2]), dtype=tf.float32)
    subst_model = HKY()
    params = hky_params
    param_key = 'frequencies'
    with tf.GradientTape(persistent=True) as t:
        t.watch(params[param_key])
        q_norm = subst_model.q_norm(**params)
        transition_probs_vals = transition_probs_expm(q_norm, category_rates, branch_lengths)
    tf_jacs = t.jacobian(transition_probs_vals, params[param_key], experimental_use_pfor=False)
    transition_probs_inv = tf.linalg.inv(transition_probs_vals)

    eigen = subst_model.eigen(**params)

    q_diffs = subst_model.q_norm_frequency_differentials(**params)[freq_index]
    diffs = transition_probs_differential(q_diffs, eigen, branch_lengths, category_rates, inv_mult=inv_mult)
    tf_diffs = (transition_probs_inv @ tf_jacs[:, :, :, :, freq_index]) if inv_mult else tf_jacs[:, :, :, :, freq_index]

    assert_allclose(tf_diffs.numpy(), diffs.numpy())

def test_transition_prob_differential_freq_num(hky_params, single_rates):
    import numdifftools

    branch_lengths = tf.convert_to_tensor(np.array([0.1]), dtype=tf.float32)
    subst_model = HKY()
    params = hky_params
    freq_index = 2

    eigen = subst_model.eigen(**params)

    freq_vals = hky_params['frequencies'].numpy()
    freq_val = freq_vals[freq_index]

    def transition_prob_func(freq):
        freq_vals[freq_index] = freq
        q_norm = subst_model.q_norm(freq_vals, kappa=hky_params['kappa'])
        return transition_probs_expm(q_norm, single_rates, branch_lengths)[0, 0].numpy()

    num_diffs = numdifftools.Jacobian(transition_prob_func)(freq_val).T
    q_diffs = subst_model.q_norm_frequency_differentials(**params)[freq_index]
    diffs = transition_probs_differential(q_diffs, eigen, branch_lengths, single_rates, inv_mult=False)

    assert_allclose(num_diffs, diffs[0, 0].numpy())






