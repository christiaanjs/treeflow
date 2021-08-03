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
        self.postorder_partials_ta = tf.TensorArray(
            dtype=DEFAULT_FLOAT_DTYPE_TF,
            size=self.get_vertex_count(),
            element_shape=self.leaf_partials.shape[1:],
        )
        for i in range(self.taxon_count):
            self.postorder_partials_ta = self.postorder_partials_ta.write(
                i, self.leaf_partials[i]
            )

        node_indices = tf.reshape(self.node_indices_tensor, [-1, 1, 1])
        child_transition_probs = tf.gather(transition_probs, self.child_indices_tensor)

        for i in range(self.taxon_count - 1):
            node_index = self.node_indices_tensor[i]
            node_child_transition_probs = child_transition_probs[
                i
            ]  # child, ..., parent character, child character
            node_child_indices = self.child_indices_tensor[i]
            child_partials = self.postorder_partials_ta.gather(
                node_child_indices
            )  # Child, ..., category, pattern, child character
            parent_child_probs = tf.expand_dims(
                node_child_transition_probs, -4
            ) * tf.expand_dims(  # child, ..., category, pattern, parent char, child char
                child_partials, -2
            )
            node_partials = tf.reduce_prod(
                tf.reduce_sum(
                    parent_child_probs,
                    axis=-1,
                ),
                axis=0,
            )
            self.postorder_partials_ta = self.postorder_partials_ta.write(
                node_index, node_partials
            )

    def compute_likelihood_from_partials(self, freqs, category_weights):
        root_partials = self.postorder_partials_ta.gather([2 * self.taxon_count - 2])[0]
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

    def init_preorder_partials(
        self, frequencies
    ):  # Node, ..., category, pattern, character
        self.root_preorder_partials = tf.broadcast_to(
            tf.expand_dims(tf.expand_dims(frequencies, -2), -2),
            self.leaf_partials.shape[1:],
        )

    def compute_preorder_partials(self, transition_probs):
        self.postorder_partials = self.postorder_partials_ta.stack()
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

        self.preorder_partials_ta = tf.TensorArray(
            dtype=DEFAULT_FLOAT_DTYPE_TF,
            size=self.get_vertex_count(),
            element_shape=self.leaf_partials.shape[1:],
        )

        self.preorder_partials_ta.write(
            self.get_vertex_count() - 1, self.root_preorder_partials
        )

        for i in range(self.get_vertex_count() - 1):
            node_index = self.preorder_indices_tensor[i]
            parent_index = self.preorder_parent_indices_tensor[i]
            parent_partials = self.preorder_partials_ta.gather([parent_index])[0]
            parent_prods = parent_partials * sibling_sums[i]
            node_transition_probs = preorder_transition_probs[i]
            node_partials = tf.reduce_sum(
                tf.expand_dims(node_transition_probs, 0)
                * tf.expand_dims(parent_prods, 3),
                axis=-2,
            )
            self.preorder_partials_ta = self.preorder_partials_ta.write(
                node_index, node_partials
            )

        self.preorder_partials = self.preorder_partials_ta.stack()

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
