import pytest
import tensorflow as tf
import numpy as np
from pathlib import Path

import treeflow.sequences
import treeflow.substitution_model
import treeflow.tensorflow_likelihood
import treeflow.tree_processing

def our_convert_to_tensor(x):
    return tf.convert_to_tensor(np.array(x), dtype=tf.float32)

def single_hky_params_():
    return {
      'frequencies': our_convert_to_tensor([0.23, 0.27, 0.24, 0.26]),
      'kappa': our_convert_to_tensor(2.0)
    }

@pytest.fixture
def single_hky_params():
    return single_hky_params_()

@pytest.fixture(params=[
    single_hky_params_(),
    {
        'frequencies': our_convert_to_tensor([0.4, 0.3, 0.2, 0.1]),
        'kappa': our_convert_to_tensor(10.0)
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

@pytest.fixture(params=[our_convert_to_tensor([0.1]), our_convert_to_tensor([0.4, 2.2])])
def branch_lengths(request):
    return request.param

def single_rates_():
    return our_convert_to_tensor(1.0)

@pytest.fixture
def single_rates():
    return single_rates_()

def single_weights_():
    return our_convert_to_tensor(1.0)

@pytest.fixture
def single_weights():
    return single_weights_()

def double_weights_():
    return our_convert_to_tensor([0.6, 0.4])

@pytest.fixture(params=[single_weights_(), double_weights_()])
def category_rates(request):
    return request.param

@pytest.fixture(params=[
    (single_weights_(), single_rates_()),
    (double_weights_(), our_convert_to_tensor([0.9, 1.1]))]
)
def weights_rates(request):
    return request.param

@pytest.fixture
def prep_likelihood():
    def prep_likelihood_(newick_file, fasta_file, subst_model, rates, weights, frequencies, **subst_params):
        eigendecomp = subst_model.eigen(frequencies, **subst_params)
        tf_likelihood = treeflow.tensorflow_likelihood.TensorflowLikelihood(category_count=len(rates))

        tree, taxon_names = treeflow.tree_processing.parse_newick(newick_file)

        branch_lengths = treeflow.sequences.get_branch_lengths(tree)

        tf_likelihood.set_topology(treeflow.tree_processing.update_topology_dict(tree['topology']))

        sequences, pattern_counts = treeflow.sequences.get_encoded_sequences(fasta_file, taxon_names)
        tf_likelihood.init_postorder_partials(sequences, pattern_counts)

        transition_probs = treeflow.substitution_model.transition_probs(eigendecomp, rates, branch_lengths)
        tf_likelihood.compute_postorder_partials(transition_probs)
        tf_likelihood.init_preorder_partials(frequencies)
        tf_likelihood.compute_preorder_partials(transition_probs)
        return tf_likelihood, branch_lengths, eigendecomp
    return prep_likelihood_
