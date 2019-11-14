import pytest
import tensorflow as tf
from numpy.testing import assert_allclose
from treeflow.substitution_model import JC, HKY

def assert_eigen(subst_model, **params):
    evec, eval, ivec = subst_model.eigen(**params)
    q = subst_model.q_norm(**params)
    assert_allclose(q.numpy(), (evec @ tf.linalg.diag(eval) @ ivec).numpy())

def test_jc_eigen():
    jc = JC()
    assert_eigen(jc, frequencies=jc.frequencies())

def test_hky_eigen(hky_params):
    assert_eigen(HKY(), **hky_params)

# TODO: test_gtr_eigen

def assert_freq_differentials(subst_model, frequencies, **params):
    with tf.GradientTape() as t:
        t.watch(frequencies)
        q = subst_model.q(frequencies, **params)
    tf_jac = t.jacobian(q, frequencies)
    tf_diffs = tf.transpose(tf_jac, perm=[2, 0, 1]).numpy()
    diffs = subst_model.q_frequency_differentials(frequencies, **params)
    assert_allclose(tf_diffs, diffs)

def test_hky_frequency_differentials(hky_params):
    assert_freq_differentials(HKY(), **hky_params)

def assert_param_differentials(subst_model, param_key, **params):
    with tf.GradientTape() as t:
        t.watch(params[param_key])
        q = subst_model.q(**params)
    tf_jac = t.jacobian(q, params[param_key])
    diffs = subst_model.q_param_differentials(**params)[param_key]
    if tf.rank(diffs) > 2:
        tf_diffs = tf.transpose(tf_jac, perm=[2, 0, 1]).numpy()
    else:
        tf_diffs = tf_jac.numpy()

    assert_allclose(tf_diffs, diffs)
        
@pytest.mark.parametrize("param_key", HKY().param_keys())
def test_hky_param_differentials(hky_params, param_key):
    assert_param_differentials(HKY(), param_key, **hky_params)
