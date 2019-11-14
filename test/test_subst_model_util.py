import pytest
from treeflow.substitution_model import HKY, normalising_constant, normalised_differential, transition_probs, transition_probs_differential
import numpy as np
from numpy.testing import assert_allclose
import tensorflow as tf

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
def test_transition_prob_differential(hky_params, param_key):
    branch_lengths = tf.convert_to_tensor(np.array([0.1, 0.9, 2.2]))
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
        diffs = transition_probs_differential(q_diffs, eigen,  branch_lengths)
        tf_diffs = transition_probs_inv @ tf_jacs
    else:
        q_diffs = subst_model.q_norm_frequency_differentials(**params)
        diffs = tf.stack([transition_probs_differential(q_diffs[i], eigen, branch_lengths) for i in range(4)])
        tf_diffs = tf.expand_dims(transition_probs_inv, 0) @ tf.transpose(tf_jacs, perm=[3, 0, 1, 2])
    assert_allclose(tf_diffs.numpy(), diffs.numpy())

