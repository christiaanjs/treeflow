from tensorflow_likelihood import init_partials_dict, parse_fasta, compress_sites, TensorflowLikelihood
import tensorflow as tf
import numpy as np

class TensorflowScanLikelihood(TensorflowLikelihood):
    def __init__(self, *args, **kwargs):
        super(TensorflowScanLikelihood, self).__init__(*args, **kwargs)
        self.node_indices_tensor = tf.convert_to_tensor(self.postorder_node_indices)
        self.child_indices_tensor = tf.convert_to_tensor(self.child_indices[self.postorder_node_indices])
        preorder_node_indices = self.get_preorder_traversal_indices()[1:]
        self.preorder_node_indices_tensor = tf.convert_to_tensor(preorder_node_indices)
        self.preorder_sibling_indices_tensor = tf.convert_to_tensor(self.get_sibling_indices()[preorder_node_indices])
        self.preorder_parent_indices_tensor = tf.convert_to_tensor(self.get_parent_indices()[preorder_node_indices])

    def init_postorder_partials(self, fasta_file):
        super(TensorflowScanLikelihood, self).init_postorder_partials(fasta_file)
        for node_index in self.postorder_node_indices:
            self.postorder_partials[node_index] = np.zeros((len(self.pattern_counts), 4))
        self.postorder_partials = tf.convert_to_tensor(np.stack(self.postorder_partials))

    def compute_postorder_partials(self, transition_probs):
        node_indices = tf.reshape(self.node_indices_tensor, [-1, 1, 1])
        child_transition_probs =  tf.gather(transition_probs, self.child_indices_tensor)
        def do_integration(partials, elems):
            node_index, node_child_transition_probs, node_child_indices = elems
            child_partials = tf.gather(partials, node_child_indices)
            node_partials = tf.reduce_prod(tf.reduce_sum(tf.expand_dims(node_child_transition_probs, 1) * tf.expand_dims(child_partials, 2), axis=3), axis=0)
            return tf.tensor_scatter_nd_update(partials, node_index, tf.expand_dims(node_partials, axis=0))
        self.postorder_partials = tf.scan(do_integration, (node_indices, child_transition_probs, self.child_indices_tensor), self.postorder_partials)[-1]
    
    def init_preorder_partials(self, frequencies):
        zeros = tf.zeros([self.get_vertex_count(), len(self.pattern_counts), 4], dtype=tf.float64)
        self.preorder_partials = tf.tensor_scatter_nd_update(
            zeros,
            np.array([[self.get_vertex_count() - 1]]),
            tf.expand_dims(tf.broadcast_to(tf.expand_dims(frequencies, 0), (len(self.pattern_counts), 4)), 0))

    def compute_preorder_partials(self, transition_probs):
        node_indices = tf.reshape(self.preorder_node_indices_tensor, [-1, 1, 1])
        preorder_transition_probs = tf.gather(transition_probs, self.preorder_node_indices_tensor)
        sibling_transition_probs = tf.gather(transition_probs, self.preorder_sibling_indices_tensor)
        sibling_postorder_partials = tf.gather(self.postorder_partials, self.preorder_sibling_indices_tensor)
        sibling_sums = tf.reduce_sum(tf.expand_dims(sibling_transition_probs, 1) * tf.expand_dims(sibling_postorder_partials, 2), axis=3)
        def do_integration(partials, elems):
            node_index, node_sibling_sums, node_transition_probs, node_parent_index = elems
            parent_partials = partials[node_parent_index]
            parent_prods = parent_partials * node_sibling_sums
            node_partials = tf.reduce_sum(tf.expand_dims(node_transition_probs, 0) * tf.expand_dims(parent_prods, 2), axis=1)
            return tf.tensor_scatter_nd_update(partials, node_index, tf.expand_dims(node_partials, axis=0))
        self.preorder_partials = tf.scan(do_integration, (node_indices, sibling_sums, preorder_transition_probs, self.preorder_parent_indices_tensor), self.preorder_partials)[-1]

    def compute_edge_derivatives(self, differential_matrices):
        differential_transpose = tf.transpose(differential_matrices, perm=[0, 2, 1])
        site_likelihoods = tf.reduce_sum(self.postorder_partials[-1] * self.preorder_partials[-1], axis=1)
        site_derivatives = tf.reduce_sum(
            tf.expand_dims(self.postorder_partials[:-1], 3) *
                tf.expand_dims(differential_transpose, 1) *
                tf.expand_dims(self.preorder_partials[:-1], 2),
            axis=[2,3]
        )
        return tf.reduce_sum(tf.expand_dims(self.pattern_counts / site_likelihoods, 0) * site_derivatives, axis=1)
    
    def compute_branch_length_derivatives(self, q):
        return self.compute_edge_derivatives(tf.expand_dims(tf.transpose(q), 0))
