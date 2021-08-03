import tensorflow as tf
import numpy as np
import treeflow.substitution_model
from treeflow import DEFAULT_FLOAT_DTYPE_TF

DEFAULT_INT_DTYPE_TF = tf.int32  # int32 required for TensorArray.gather


class TensorflowLikelihood:
    def __init__(self, category_count=1, *args, **kwargs):
        self.category_count = category_count

    def set_topology(self, topology_dict):
        self.taxon_count = len(topology_dict["postorder_node_indices"]) + 1
        self.node_indices_tensor = tf.convert_to_tensor(
            topology_dict["postorder_node_indices"], dtype=DEFAULT_INT_DTYPE_TF
        )
        self.child_indices_tensor = tf.cast(
            tf.gather(
                topology_dict["child_indices"], topology_dict["postorder_node_indices"]
            ),
            dtype=DEFAULT_INT_DTYPE_TF,
        )

        preorder_indices = topology_dict["preorder_indices"][1:]
        self.preorder_indices_tensor = tf.convert_to_tensor(
            preorder_indices, dtype=DEFAULT_INT_DTYPE_TF
        )
        self.preorder_sibling_indices_tensor = tf.cast(
            tf.gather(
                topology_dict["sibling_indices"],
                preorder_indices,
            ),
            dtype=DEFAULT_INT_DTYPE_TF,
        )
        self.preorder_parent_indices_tensor = tf.cast(
            tf.gather(
                topology_dict["parent_indices"],
                preorder_indices,
            ),
            dtype=DEFAULT_INT_DTYPE_TF,
        )

    def get_vertex_count(self):
        return 2 * self.taxon_count - 1

    def init_postorder_partials(self, sequences_encoded, pattern_counts=None):
        """
        Sequence shape:
        # Taxon, ..., pattern, character
        Partial shape:
        # Node, ..., category, pattern, character
        """
        self.taxon_count = sequences_encoded.shape[0]
        self.pattern_count = sequences_encoded.shape[-2]
        self.pattern_counts = (
            tf.ones([self.pattern_count]) if pattern_counts is None else pattern_counts
        )
        self.batch_shape = sequences_encoded.shape[1:-2]
        character_shape = sequences_encoded.shape[-1]
        self.leaf_partials = tf.broadcast_to(
            tf.expand_dims(sequences_encoded, -2),  # Add category
            [self.taxon_count]
            + self.batch_shape
            + [self.pattern_count, self.category_count, character_shape],
        )

    def compute_postorder_partials(self, transition_probs):
        node_indices = tf.reshape(self.node_indices_tensor, [-1, 1, 1])
        child_transition_probs = tf.gather(transition_probs, self.child_indices_tensor)

        partials = tf.TensorArray(
            dtype=DEFAULT_FLOAT_DTYPE_TF,
            size=self.get_vertex_count(),
            element_shape=self.leaf_partials.shape[1:],
        )
        for i in range(self.taxon_count):
            partials = partials.write(i, self.leaf_partials[i])

        for i in range(self.taxon_count - 1):
            node_index = self.node_indices_tensor[i]
            node_child_transition_probs = child_transition_probs[
                i
            ]  # child, ..., parent character, child character
            node_child_indices = self.child_indices_tensor[i]
            child_partials = partials.gather(
                node_child_indices
            )  # Child, ..., pattern, child character
            parent_child_probs = tf.expand_dims(
                node_child_transition_probs, -3
            ) * tf.expand_dims(  # child, ..., pattern, parent char, child char
                child_partials, -2
            )
            node_partials = tf.reduce_prod(
                tf.reduce_sum(
                    parent_child_probs,
                    axis=-1,
                ),
                axis=0,
            )
            partials = partials.write(node_index, node_partials)
        self.postorder_partials = partials

    def compute_likelihood_from_partials(self, freqs, category_weights):
        root_partials = self.postorder_partials.gather([2 * self.taxon_count - 2])[0]
        cat_likelihoods = tf.reduce_sum(freqs * root_partials, axis=-1)
        site_likelihoods = tf.reduce_sum(category_weights * cat_likelihoods, axis=-1)
        return tf.reduce_sum(
            tf.math.log(site_likelihoods) * self.pattern_counts, axis=-1
        )

    def compute_likelihood(
        self, branch_lengths, category_rates, category_weights, freqs, eigendecomp
    ):
        transition_probs = treeflow.substitution_model.transition_probs(
            eigendecomp, category_rates, branch_lengths
        )
        self.compute_postorder_partials(transition_probs)
        return self.compute_likelihood_from_partials(freqs, category_weights)

    def compute_likelihood_expm(
        self, branch_lengths, category_rates, category_weights, freqs, q
    ):
        transition_probs = treeflow.substitution_model.transition_probs_expm(
            q, category_rates, branch_lengths
        )
        self.compute_postorder_partials(transition_probs)
        return self.compute_likelihood_from_partials(freqs, category_weights)

    def init_preorder_partials(self, frequencies):
        return
        zeros = tf.zeros(
            [self.get_vertex_count(), len(self.pattern_counts), self.category_count, 4],
            dtype=DEFAULT_FLOAT_DTYPE_TF,
        )
        self.preorder_partials = tf.tensor_scatter_nd_update(
            zeros,
            np.array([[self.get_vertex_count() - 1]]),
            tf.expand_dims(
                tf.broadcast_to(
                    tf.reshape(frequencies, [1, 1, 4]),
                    (len(self.pattern_counts), self.category_count, 4),
                ),
                0,
            ),
        )

    def compute_preorder_partials(self, transition_probs):
        return
        node_indices = tf.reshape(self.preorder_indices_tensor, [-1, 1, 1])
        preorder_transition_probs = tf.gather(
            transition_probs, self.preorder_indices_tensor
        )
        sibling_transition_probs = tf.gather(
            transition_probs, self.preorder_sibling_indices_tensor
        )
        sibling_postorder_partials = tf.gather(
            self.postorder_partials, self.preorder_sibling_indices_tensor
        )
        sibling_sums = tf.reduce_sum(
            tf.expand_dims(sibling_transition_probs, 1)
            * tf.expand_dims(sibling_postorder_partials, 3),
            axis=4,
        )

        def do_integration(partials, elems):
            (
                node_index,
                node_sibling_sums,
                node_transition_probs,
                node_parent_index,
            ) = elems
            parent_partials = partials[node_parent_index]
            parent_prods = parent_partials * node_sibling_sums
            node_partials = tf.reduce_sum(
                tf.expand_dims(node_transition_probs, 0)
                * tf.expand_dims(parent_prods, 3),
                axis=2,
            )
            return tf.tensor_scatter_nd_update(
                partials, node_index, tf.expand_dims(node_partials, axis=0)
            )

        self.preorder_partials = tf.scan(
            do_integration,
            (
                node_indices,
                sibling_sums,
                preorder_transition_probs,
                self.preorder_parent_indices_tensor,
            ),
            self.preorder_partials,
        )[-1]

    def compute_cat_derivatives(self, differential_matrices, sum_branches=False):
        differential_transpose = tf.transpose(differential_matrices, perm=[0, 1, 3, 2])
        return tf.reduce_sum(
            tf.expand_dims(self.postorder_partials[:-1], 4)
            * tf.expand_dims(differential_transpose, 1)
            * tf.expand_dims(self.preorder_partials[:-1], 3),
            axis=([0, 3, 4] if sum_branches else [3, 4]),
        )

    def compute_site_derivatives(
        self, differential_matrices, category_weights, sum_branches=False
    ):
        cat_derivatives = self.compute_cat_derivatives(
            differential_matrices, sum_branches=sum_branches
        )
        return tf.reduce_sum(cat_derivatives * category_weights, axis=-1)

    def compute_cat_likelihoods(self):
        return tf.reduce_sum(
            self.postorder_partials[-1] * self.preorder_partials[-1], axis=-1
        )

    def compute_site_likelihoods(self, category_weights):
        return tf.reduce_sum(self.compute_cat_likelihoods() * category_weights, axis=-1)

    def compute_derivative(self, differential_matrices, category_weights):
        site_derivatives = self.compute_site_derivatives(
            differential_matrices, category_weights, sum_branches=True
        )
        site_coefficients = self.pattern_counts / self.compute_site_likelihoods(
            category_weights
        )
        return tf.reduce_sum(site_derivatives * site_coefficients)

    def compute_edge_derivatives(self, differential_matrices, category_weights):
        site_likelihoods = self.compute_site_likelihoods(category_weights)
        site_derivatives = self.compute_site_derivatives(
            differential_matrices, category_weights
        )
        return tf.reduce_sum(
            self.pattern_counts / site_likelihoods * site_derivatives, axis=-1
        )

    def compute_branch_length_derivatives(self, q, category_rates, category_weights):
        site_coefficients = self.pattern_counts / self.compute_site_likelihoods(
            category_weights
        )
        cat_derivatives = (
            self.compute_cat_derivatives(tf.reshape(q, [1, 1, 4, 4])) * category_rates
        )  # TODO: Should we multiply differential matrices by rates?
        site_derivatives = tf.reduce_sum(cat_derivatives * category_weights, axis=-1)
        return tf.reduce_sum(site_coefficients * site_derivatives, axis=-1)

    def compute_rate_derivatives(self, q, branch_lengths, category_weights):
        site_coefficients = self.pattern_counts / self.compute_site_likelihoods(
            category_weights
        )
        dist_derivatives = self.compute_cat_derivatives(tf.reshape(q, [1, 1, 4, 4]))
        site_derivatives = tf.reduce_sum(
            dist_derivatives * tf.reshape(branch_lengths, [-1, 1, 1]), axis=0
        )
        return (
            tf.reduce_sum(
                site_derivatives * tf.expand_dims(site_coefficients, axis=1), axis=0
            )
            * category_weights
        )

    def compute_weight_derivatives(self, category_weights):
        site_coefficients = self.pattern_counts / self.compute_site_likelihoods(
            category_weights
        )
        return tf.reduce_sum(
            self.compute_cat_likelihoods() * tf.expand_dims(site_coefficients, axis=1),
            axis=0,
        )

    def compute_frequency_derivative(
        self, differential_matrices, frequency_index, category_weights
    ):
        site_likelihoods = self.compute_site_likelihoods(category_weights)
        cat_derivatives = self.compute_cat_derivatives(
            differential_matrices, sum_branches=True
        )
        site_coefficients = self.pattern_counts / site_likelihoods
        return tf.reduce_sum(
            site_coefficients
            * tf.reduce_sum(
                (cat_derivatives + self.postorder_partials[-1, :, :, frequency_index])
                * category_weights,
                axis=-1,
            )
        )
