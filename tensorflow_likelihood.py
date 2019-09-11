import tensorflow as tf
import numpy as np
import re
from collections import Counter
from base_likelihood import BaseLikelihood

init_partials_dict = {
    'A':[1.,0.,0.,0.],
    'G':[0.,1.,0.,0.],
    'C':[0.,0.,1.,0.],
    'T':[0.,0.,0.,1.],
    '-':[1.,1.,1.,1.],
    '?':[1.,1.,1.,1.],
    'N':[1.,1.,1.,1.],
    'R':[1.,1.,0.,0.],
    'Y':[0.,0.,1.,1.],
    'S':[0.,1.,1.,0.], 
    'W':[1.,0.,0.,1.],
    'K':[0.,1.,0.,1.],
    'M':[1.,0.,1.,0.],
    'B':[0.,1.,1.,1.],
    'D':[1.,1.,0.,1.],
    'H':[1.,0.,1.,1.],
    'V':[1.,1.,1.,0.],
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
    taxa = list(sequence_dict.keys())
    sequences = [sequence_dict[taxon] for taxon in taxa]
    patterns = list(zip(*sequences)) 
    count_dict = Counter(patterns)
    pattern_ordering = list(count_dict.keys())
    patterns = list(zip(*patterns))
    counts = [count_dict[pattern] for pattern in pattern_ordering]
    pattern_dict = dict(zip(taxa, patterns))
    return pattern_dict, counts

class TensorflowLikelihood(BaseLikelihood):
    def __init__(self, fasta_file, *args, **kwargs):
        super(TensorflowLikelihood, self).__init__(fasta_file=fasta_file, *args, **kwargs)
        self.child_indices = self.get_child_indices()
        self.postorder_node_indices = self.get_postorder_node_traversal_indices()
        print(self.postorder_node_indices)

        self.init_partials(fasta_file)

    def init_partials(self, fasta_file):
        newick = self.inst.tree_collection.newick()
        leaf_names = re.findall(r'(\w+)(?=:)', newick)
        sequence_dict = parse_fasta(fasta_file)
        pattern_dict, self.pattern_counts = compress_sites(sequence_dict) 
        self.partials = [None] * self.get_vertex_count()

        for leaf_index in range(len(leaf_names)):
            self.partials[leaf_index] = np.zeros((len(self.pattern_counts), 4))
            for pattern_index in range(len(self.pattern_counts)):
                self.partials[leaf_index][pattern_index] = init_partials_dict[pattern_dict[leaf_names[leaf_index]][pattern_index]]

    def compute_likelihood(self, branch_lengths):
        raise NotImplementedError()
