from __future__ import annotations
import pytest
import tensorflow as tf


@pytest.fixture
def hky_params(tensor_constant):
    return dict(
        frequencies=tensor_constant([0.23, 0.27, 0.24, 0.26]),
        kappa=tensor_constant(2.0),
    )


@pytest.fixture
def hky_params_vec(hky_params, hello_tensor_tree):
    kappa = hky_params["kappa"]
    kappa_vec = kappa + tf.cast(
        tf.range(hello_tensor_tree.branch_lengths.shape), kappa.dtype
    )
    frequencies = hky_params["frequencies"]
    frequencies_b = tf.broadcast_to(frequencies, kappa_vec.shape + frequencies.shape)
    return dict(frequencies=frequencies_b, kappa=kappa_vec)


@pytest.fixture
def hello_hky_log_likelihood():
    return -88.86355638556158
