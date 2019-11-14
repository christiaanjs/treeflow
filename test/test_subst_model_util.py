import pytest
from treeflow.substitution_model import HKY, normalising_constant, normalised_differential
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
        diffs = subst_model.q_param_differentials(**params)[param_key]
        norm_diffs = normalised_differential(diffs, q_norm, norm_const, params['frequencies']).numpy()
        tf_diffs = tf_jac.numpy()
    else:
        diffs = subst_model.q_frequency_differentials(**params)
        norm_diffs = np.stack([normalised_differential(diffs[i], q_norm, norm_const, params['frequencies'], frequency_index=i, q=q).numpy() for i in range(4)])
        tf_diffs = tf.transpose(tf_jac, perm=[2, 0, 1]).numpy()
    assert_allclose(tf_diffs, norm_diffs)

