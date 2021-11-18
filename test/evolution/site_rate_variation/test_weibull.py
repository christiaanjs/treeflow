from treeflow.evolution.site_rate_variation.weibull import WeibullSiteModel
import numpy as np
import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF

# TODO: More comprehensive tests
def test_WeibullSiteModel():
    model = WeibullSiteModel(4)
    rates, weights = model.rates_weights(
        lambd=tf.constant(2.0, dtype=DEFAULT_FLOAT_DTYPE_TF),
        k=tf.constant(2.0, dtype=DEFAULT_FLOAT_DTYPE_TF),
    )
    assert np.isfinite(rates.numpy()).all()
    assert np.isfinite(weights.numpy()).all()
