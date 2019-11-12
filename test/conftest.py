import pytest
import tensorflow as tf
import numpy as np

@pytest.fixture(params=[
    {

        'frequencies': tf.convert_to_tensor(np.array([0.23, 0.27, 0.24, 0.26])),
        'kappa': tf.convert_to_tensor(np.array(2.0))
    },
    {
        'frequencies': tf.convert_to_tensor(np.array([0.4, 0.3, 0.2, 0.1])),
        'kappa': tf.convert_to_tensor(np.array(10.0))
    }
])

def hky_params(request):
    return request.param
