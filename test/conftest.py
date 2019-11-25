import pytest
import tensorflow as tf
import numpy as np
from pathlib import Path

def single_hky_params_():
    return {
      'frequencies': tf.convert_to_tensor(np.array([0.23, 0.27, 0.24, 0.26])),
      'kappa': tf.convert_to_tensor(np.array(2.0))
    }

@pytest.fixture
def single_hky_params():
    return single_hky_params_()

@pytest.fixture(params=[
    single_hky_params_(),
    {
        'frequencies': tf.convert_to_tensor(np.array([0.4, 0.3, 0.2, 0.1])),
        'kappa': tf.convert_to_tensor(np.array(10.0))
    }
])
def hky_params(request):
    return request.param

@pytest.fixture(params=list(range(4)))
def freq_index(request):
    return request.param

data_dir = Path('data')

@pytest.fixture
def hello_newick_file():
    return str(data_dir / 'hello.nwk')

@pytest.fixture
def hello_fasta_file():
    return str(data_dir / 'hello.fasta')

@pytest.fixture(params=[tf.convert_to_tensor(np.array([0.1])), tf.convert_to_tensor(np.array([0.4, 2.2]))])
def branch_lengths(request):
    return request.param

def single_rates_():
    return tf.convert_to_tensor(np.array([1.0]))

@pytest.fixture
def single_rates():
    return single_rates_()

@pytest.fixture(params=[single_rates_(), tf.convert_to_tensor(np.array([0.9, 1.1]))])
def category_rates(request):
    return request.param

def single_weights_():
    return tf.convert_to_tensor(np.array([1.0]))

@pytest.fixture
def single_weights():
    return single_weights_()

@pytest.fixture(params=[single_weights_(), tf.convert_to_tensor(np.array([0.6, 0.4]))])
def category_weights(request):
    return request.param
