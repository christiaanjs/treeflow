from treeflow.evolution.site_rate_variation.gamma import GammaSiteModel
import numpy as np
import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF

# TODO: More comprehensive tests
def test_GammaSiteModel():
    model = GammaSiteModel(4)
    rates, weights = model.rates_weights(
        a=tf.constant(2.0, dtype=DEFAULT_FLOAT_DTYPE_TF)
    )
    assert np.isfinite(rates.numpy()).all()
    assert np.isfinite(weights.numpy()).all()
