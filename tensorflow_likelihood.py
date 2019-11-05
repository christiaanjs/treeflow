import tensorflow as tf
import numpy as np
import re
from collections import Counter
from base_likelihood import BaseLikelihood
import substitution_model

init_partials_dict = {
    'A':[1.,0.,0.,0.],
    'C':[0.,1.,0.,0.],
    'G':[0.,0.,1.,0.],
    'T':[0.,0.,0.,1.],
    '-':[1.,1.,1.,1.],
    '?':[1.,1.,1.,1.],
    'N':[1.,1.,1.,1.],
    #'R':[1.,1.,0.,0.],# TODO: Fix indexing of these
    #'Y':[0.,0.,1.,1.],
    #'S':[0.,1.,1.,0.], 
    #'W':[1.,0.,0.,1.],
    #'K':[0.,1.,0.,1.],
    #'M':[1.,0.,1.,0.],
    #'B':[0.,1.,1.,1.],
    #'D':[1.,1.,0.,1.],
    #'H':[1.,0.,1.,1.],
    #'V':[1.,1.,1.,0.],
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

class TensorflowLikelihood(BaseLikelihood):
    def __init__(self, fasta_file, *args, **kwargs):
        super(TensorflowLikelihood, self).__init__(fasta_file=fasta_file, *args, **kwargs)
        self.child_indices = self.get_child_indices()
        self.postorder_node_indices = self.get_postorder_node_traversal_indices()

        self.init_postorder_partials(fasta_file)

    def init_postorder_partials(self, fasta_file):
        newick = self.inst.tree_collection.newick()
        leaf_names = re.findall(r'(\w+)(?=:)', newick)
        sequence_dict = parse_fasta(fasta_file)
        pattern_dict, self.pattern_counts = compress_sites(sequence_dict) 
        self.postorder_partials = [None] * self.get_vertex_count()

        for leaf_index in range(len(leaf_names)):
            self.postorder_partials[leaf_index] = np.zeros((len(self.pattern_counts), 4))
            for pattern_index in range(len(self.pattern_counts)):
                self.postorder_partials[leaf_index][pattern_index] = init_partials_dict[pattern_dict[leaf_names[leaf_index]][pattern_index]]

    def compute_postorder_partials(self, transition_probs):
        for node_index in self.postorder_node_indices:
            child_indices = self.child_indices[node_index]
            child_partials = tf.stack([self.postorder_partials[child_index] for child_index in child_indices], axis=1) 
            child_transition_probs = tf.gather(transition_probs, child_indices)
            self.postorder_partials[node_index] = tf.reduce_prod(tf.reduce_sum(tf.expand_dims(child_transition_probs, 0) * tf.expand_dims(child_partials, 2), axis=3), axis=1)

    @tf.function
    def compute_likelihood_tf(self, branch_lengths, freqs, eigendecomp):
        transition_probs = substitution_model.transition_probs(eigendecomp, branch_lengths)
        self.compute_postorder_partials(transition_probs)
        site_likelihoods = tf.reduce_sum(tf.expand_dims(freqs, 0) * self.postorder_partials[-1], axis=1)
        return tf.reduce_sum(tf.math.log(site_likelihoods) * self.pattern_counts)

    def compute_likelihood(self, branch_lengths, freqs, eigendecomp):
        return self.compute_likelihood_tf(branch_lengths, freqs, eigendecomp).numpy()
    
    @tf.function
    def compute_gradient_tf(self, branch_lengths, freqs, eigendecomp, *wrt):
        return tf.gradients(self.compute_likelihood_tf(branch_lengths, freqs, eigendecomp), [branch_lengths] + list(wrt))

    def compute_gradient(self, branch_lengths, freqs, eigendecomp, *wrt):
        return [x.numpy() for x in self.compute_gradient_tf(branch_lengths, freqs, eigendecomp, *wrt)]

class TensorflowTwoPassLikelihood(TensorflowLikelihood):
    def __init__(self, *args, **kwargs):
        super(TensorflowTwoPassLikelihood, self).__init__(*args, **kwargs)
        self.preorder_indices = self.get_preorder_traversal_indices()
        self.parent_indices = self.get_parent_indices()
        self.sibling_indices = self.get_sibling_indices()
        self.init_preorder_partials(substitution_model.jc_frequencies)

    def init_preorder_partials(self, frequencies):
        self.preorder_partials = [None] * self.get_vertex_count()
        self.parent_prods = [None] * self.get_vertex_count() # Root is empty
        self.preorder_partials[-1] = tf.broadcast_to(tf.expand_dims(frequencies, 0), (len(self.pattern_counts), 4))

    def compute_preorder_partials(self, transition_probs):
        for node_index in self.preorder_indices[1:]:
            sibling_index = self.sibling_indices[node_index]
            sibling_sum = tf.reduce_sum(tf.expand_dims(transition_probs[sibling_index], 0) * tf.expand_dims(self.postorder_partials[sibling_index], 1), axis=2) # TODO: Cache this in preorder traversal?
            self.parent_prods[node_index] = self.preorder_partials[self.parent_indices[node_index]] * sibling_sum
            self.preorder_partials[node_index] = tf.reduce_sum(tf.expand_dims(transition_probs[node_index], 0) * tf.expand_dims(self.parent_prods[node_index], 2), axis=1)
    
    @tf.function
    def compute_likelihood_tf(self, branch_lengths):
        transition_probs = substitution_model.transition_probs(substitution_model.jc_eigendecomposition, branch_lengths)
        @tf.custom_gradient
        def site_likelihoods(transition_probs):
            self.compute_postorder_partials(transition_probs)
            site_likelihoods = tf.reduce_sum(tf.expand_dims(substitution_model.jc_frequencies, 0) * self.postorder_partials[-1], axis=1)
            def grad(dsite_likelihoods):
                self.compute_preorder_partials(transition_probs) 
                parent_prods = tf.stack(self.parent_prods[:-1], axis=1)
                postorder_partials = tf.stack(self.postorder_partials[:-1], axis=1)
                grads = tf.expand_dims(parent_prods, axis=3) * tf.expand_dims(postorder_partials, axis=2)
                return tf.reduce_sum(tf.reshape(dsite_likelihoods, [-1, 1, 1, 1]) * grads, axis=0)
            return site_likelihoods, grad
        
        return tf.reduce_sum(tf.math.log(site_likelihoods(transition_probs)) * self.pattern_counts)


         

