import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from collections import Counter

init_partials_dict = {
    'A':[1.,0.,0.,0.],
    'C':[0.,1.,0.,0.],
    'G':[0.,0.,1.,0.],
    'T':[0.,0.,0.,1.],
    '-':[1.,1.,1.,1.],
    '?':[1.,1.,1.,1.],
    'N':[1.,1.,1.,1.],
    '.':[1.,1.,1.,1.],
    'U':[0.,0.,0.,1.]
}

def parse_fasta(filename):
    f = open(filename)
    x = f.read()
    f.close()
    def process_block(block):
        lines = block.split('\n')
        return lines[0], ''.join(lines[1:])
    return dict([process_block(block) for block in x.split('>')[1:]])

def compress_sites(sequence_dict):
    taxa = sorted(list(sequence_dict.keys()))
    sequences = [sequence_dict[taxon] for taxon in taxa]
    patterns = list(zip(*sequences)) 
    count_dict = Counter(patterns)
    pattern_ordering = sorted(list(count_dict.keys()))
    compressed_sequences = list(zip(*pattern_ordering))
    counts = [count_dict[pattern] for pattern in pattern_ordering]
    pattern_dict = dict(zip(taxa, compressed_sequences))
    return pattern_dict, counts

def encode_sequence_dict(sequence_dict, taxon_names):
    return tf.convert_to_tensor(np.array([[init_partials_dict[char] for char in sequence_dict[taxon_name]] for taxon_name in taxon_names]))


def get_branch_lengths(heights, topology):
    pass # TODO

def log_prob_conditioned(value, topology):
    patterns = value['sequences']
    pattern_counts = value['weights']

    def log_prob(heights, subst_model, frequencies, category_weights, category_rates, **subst_model_params):
        subst_model_param_keys = list(subst_model_params.keys())
        @tf.custom_gradient
        def log_prob_flat(branch_lengths, frequencies, category_weights, category_rates, *subst_model_params):
            def grad(dbranch_lengths, dfrequencies, dcategory_weights, dcategory_rates, *dsubst_model_params):
                pass # TODO
        return log_prob_flat(get_branch_lengths(heights, topology), frequencies, category_weights, category_rates,
            *[subst_model_params[key] for key in subst_model_param_keys])
    return log_prob
    

class LeafSequences(tfp.distributions.Distribution):
    def __init__(self, tree, subst_model, frequencies, category_weights, category_rates,
        validate_args=False, allow_nan_stats=True, **subst_model_params):
        super(LeafSequences, self).__init__(
            dtype={ 'sequences': tf.int64, 'weights': tf.int64 },
            reparameterization_type=tfp.distributions.NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats, 
            parameters=dict(locals()))
        self.tree = tree
        self.subst_model = subst_model
        self.frequencies = frequencies
        self.category_weights = category_weights
        self.category_rates = category_rates
        self.subst_model_params = subst_model_params

    def _log_prob(self, value):
        return log_prob_conditioned(value, self.tree['topology'])(self.tree['heights'], self.subst_model, self.frequencies, self.category_weights, self.category_rates, **self.subst_model_params)

    def _sample_n(self, n, seed=None):
        raise NotImplementedError('Sequence simulator not yet implemented')
