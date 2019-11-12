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
