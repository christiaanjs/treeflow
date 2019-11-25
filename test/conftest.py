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

@pytest.fixture
def branch_lengths():
    return np.array([0.4, 2.2])

@pytest.fixture
def category_rates():
    return np.array([1.1, 0.9])

data_dir = Path('data')

@pytest.fixture
def hello_newick_file():
    return str(data_dir / 'hello.nwk')

@pytest.fixture
def hello_fasta_file():
    return str(data_dir / 'hello.fasta')


@pytest.fixture
def single_rates():
    return tf.convert_to_tensor(np.array([1.0]))

@pytest.fixture
def single_weights():
    return tf.convert_to_tensor(np.array([1.0]))


    
