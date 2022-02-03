import numpy as np
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.model.approximation import get_mean_field_approximation


def test_get_mean_field_approximation():
    sample_size = 3
    constant = lambda x: tf.constant(x, dtype=DEFAULT_FLOAT_DTYPE_TF)
    model = tfd.JointDistributionNamed(
        dict(
            a=tfd.Normal(constant(0.0), constant(1.0)),
            b=lambda a: tfd.Sample(tfd.LogNormal(a, constant(1.0)), sample_size),
            obs=lambda b: tfd.Independent(
                tfd.Normal(b, constant(1.0)), reinterpreted_batch_ndims=1
            ),
        )
    )
    obs = constant([-1.1, 2.1, 0.1])
    pinned = model.experimental_pin(obs=obs)
    approximation = get_mean_field_approximation(
        pinned, init_loc=dict(a=constant(0.1)), dtype=DEFAULT_FLOAT_DTYPE_TF
    )
    sample = approximation.sample()
    model_log_prob = pinned.unnormalized_log_prob(sample)
    approx_log_prob = approximation.log_prob(sample)
    assert np.isfinite(model_log_prob.numpy())
    assert np.isfinite(approx_log_prob.numpy())
