import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from collections import Counter
import treeflow.tensorflow_likelihood
import treeflow.tree_processing
import treeflow.substitution_model
import treeflow.tf_util
from treeflow import DEFAULT_FLOAT_DTYPE_TF

init_partials_dict = {
    "A": [1.0, 0.0, 0.0, 0.0],
    "C": [0.0, 1.0, 0.0, 0.0],
    "G": [0.0, 0.0, 1.0, 0.0],
    "T": [0.0, 0.0, 0.0, 1.0],
    "U": [0.0, 0.0, 0.0, 1.0],
    "-": [1.0, 1.0, 1.0, 1.0],
    "?": [1.0, 1.0, 1.0, 1.0],
    "N": [1.0, 1.0, 1.0, 1.0],
    ".": [1.0, 1.0, 1.0, 1.0],
    # Note treating all degenerate bases as gaps to maintain agreement with BEAST.
    "B": [1.0, 1.0, 1.0, 1.0],
    "D": [1.0, 1.0, 1.0, 1.0],
    "H": [1.0, 1.0, 1.0, 1.0],
    "K": [1.0, 1.0, 1.0, 1.0],
    "M": [1.0, 1.0, 1.0, 1.0],
    "R": [1.0, 1.0, 1.0, 1.0],
    "S": [1.0, 1.0, 1.0, 1.0],
    "U": [1.0, 1.0, 1.0, 1.0],
    "V": [1.0, 1.0, 1.0, 1.0],
    "W": [1.0, 1.0, 1.0, 1.0],
    "Y": [1.0, 1.0, 1.0, 1.0],
}


def parse_fasta(filename):
    f = open(filename)
    x = f.read()
    f.close()

    def process_block(block):
        lines = block.split("\n")
        return lines[0], "".join(lines[1:])

    return dict([process_block(block) for block in x.split(">")[1:]])


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
    return tf.convert_to_tensor(
        np.array(
            [
                [init_partials_dict[char] for char in sequence_dict[taxon_name]]
                for taxon_name in taxon_names
            ]
        ),
        dtype=DEFAULT_FLOAT_DTYPE_TF,
    )


def get_encoded_sequences(fasta_file, taxon_names):
    sequence_dict = parse_fasta(fasta_file)
    pattern_dict, counts = compress_sites(sequence_dict)
    return {
        "sequences": encode_sequence_dict(pattern_dict, taxon_names),
        "weights": counts,
    }


def _get_branch_lengths_1d_flat(
    x_flat,
):  # TODO: Make this work properly with > 1 batch dim
    heights = x_flat[0]
    parent_indices = x_flat[1]
    return tf.gather(heights, parent_indices) - heights[:-1]


def get_branch_lengths(tree):
    heights = tree["heights"]
    batch_shape = tf.shape(heights)[:-1]
    node_count = tf.shape(heights)[-1]
    parent_indices = tf.broadcast_to(
        tree["topology"]["parent_indices"],
        tf.concat([batch_shape, [node_count - 1]], axis=0),
    )
    batch_dims = tf.shape(batch_shape)[0]
    x_flat = [heights, parent_indices]
    return treeflow.tf_util.vectorize_1d_if_needed(
        _get_branch_lengths_1d_flat, x_flat, batch_dims
    )


def log_prob_conditioned(value, topology, category_count, custom_gradient=True):

    likelihood = treeflow.tensorflow_likelihood.TensorflowLikelihood(
        category_count=category_count
    )
    likelihood.set_topology(treeflow.tree_processing.update_topology_dict(topology))
    likelihood.init_postorder_partials(
        value["sequences"],
        pattern_counts=(value["weights"] if "weights" in value else None),
    )

    def log_prob(
        branch_lengths,
        subst_model,
        category_weights,
        category_rates,
        frequencies,
        **subst_model_params
    ):
        subst_model_param_keys = list(subst_model_params.keys())

        def redict_params(subst_model_params):
            return dict(zip(subst_model_param_keys, subst_model_params))

        if custom_gradient:

            @tf.custom_gradient
            def log_prob_flat(
                branch_lengths,
                category_weights,
                category_rates,
                frequencies,
                *subst_model_params_list
            ):
                subst_model_params = redict_params(subst_model_params_list)
                eigendecomp = subst_model.eigen(frequencies, **subst_model_params)
                transition_probs = treeflow.substitution_model.transition_probs(
                    eigendecomp, category_rates, branch_lengths
                )
                likelihood.compute_postorder_partials(transition_probs)

                def grad(dlog_prob):
                    likelihood.init_preorder_partials(frequencies)
                    likelihood.compute_preorder_partials(transition_probs)
                    q = subst_model.q_norm(frequencies, **subst_model_params)
                    q_freq_differentials = subst_model.q_norm_frequency_differentials(
                        frequencies, **subst_model_params
                    )
                    freq_differentials = [
                        treeflow.substitution_model.transition_probs_differential(
                            q_freq_differentials[i],
                            eigendecomp,
                            branch_lengths,
                            category_rates,
                        )
                        for i in range(4)
                    ]
                    q_param_differentials = subst_model.q_norm_param_differentials(
                        frequencies, **subst_model_params
                    )
                    param_grads = [
                        likelihood.compute_derivative(
                            treeflow.substitution_model.transition_probs_differential(
                                q_param_differentials[param_key],
                                eigendecomp,
                                branch_lengths,
                                category_rates,
                            ),
                            category_weights,
                        )
                        for param_key in subst_model_param_keys
                    ]
                    return [
                        dlog_prob * grad
                        for grad in (
                            [
                                likelihood.compute_branch_length_derivatives(
                                    q, category_rates, category_weights
                                ),
                                likelihood.compute_weight_derivatives(category_weights),
                                likelihood.compute_rate_derivatives(
                                    q, branch_lengths, category_weights
                                ),
                                tf.stack(
                                    [
                                        likelihood.compute_frequency_derivative(
                                            freq_differentials[i], i, category_weights
                                        )
                                        for i in range(4)
                                    ]
                                ),
                            ]
                            + param_grads
                        )
                    ]

                log_prob_val = likelihood.compute_likelihood_from_partials(
                    frequencies, category_weights
                )
                return log_prob_val, grad  # TODO: Cache site likelihoods

        else:

            def log_prob_flat(
                branch_lengths,
                category_weights,
                category_rates,
                frequencies,
                *subst_model_params_list
            ):
                subst_model_params = redict_params(subst_model_params_list)
                eigendecomp = subst_model.eigen(frequencies, **subst_model_params)
                transition_probs = treeflow.substitution_model.transition_probs(
                    eigendecomp, category_rates, branch_lengths
                )
                likelihood.compute_postorder_partials(transition_probs)
                log_prob_val = likelihood.compute_likelihood_from_partials(
                    frequencies, category_weights
                )
                return log_prob_val

        def log_prob_1d(elems):
            branch_lengths, category_weights, category_rates, frequencies = elems[:4]
            subst_model_params_list = elems[4:]
            return log_prob_flat(
                branch_lengths,
                category_weights,
                category_rates,
                frequencies,
                *subst_model_params_list
            )

        batch_shape = branch_lengths.shape[
            :-1
        ]  # TODO: What's a better way to get batch shape?
        category_weights_b = tf.broadcast_to(
            category_weights, batch_shape + [category_count]
        )
        category_rates_b = tf.broadcast_to(
            category_rates, batch_shape + [category_count]
        )
        frequencies_b = tf.broadcast_to(frequencies, batch_shape + [4])
        subst_model_params_list = [
            subst_model_params[key] for key in subst_model_param_keys
        ]
        subst_model_params_b = [
            tf.broadcast_to(param, batch_shape) for param in subst_model_params_list
        ]
        elems = [
            branch_lengths,
            category_weights_b,
            category_rates_b,
            frequencies_b,
        ] + subst_model_params_b
        return treeflow.tf_util.vectorize_1d_if_needed(
            log_prob_1d, elems, batch_shape.rank
        )

    return log_prob


def log_prob_conditioned_branch_only(
    value,
    topology,
    category_count,
    subst_model,
    category_weights,
    category_rates,
    frequencies,
    custom_gradient=True,
    **subst_model_params
):

    likelihood = treeflow.tensorflow_likelihood.TensorflowLikelihood(
        category_count=category_count
    )
    likelihood.set_topology(treeflow.tree_processing.update_topology_dict(topology))
    likelihood.init_postorder_partials(
        value["sequences"],
        pattern_counts=(value["weights"] if "weights" in value else None),
    )

    if custom_gradient:

        @tf.custom_gradient
        def log_prob_1d(branch_lengths):
            eigendecomp = subst_model.eigen(frequencies, **subst_model_params)
            transition_probs = treeflow.substitution_model.transition_probs(
                eigendecomp, category_rates, branch_lengths
            )
            likelihood.compute_postorder_partials(transition_probs)

            @tf.function
            def grad(dlog_prob):
                likelihood.init_preorder_partials(frequencies)
                likelihood.compute_preorder_partials(transition_probs)
                q = subst_model.q_norm(frequencies, **subst_model_params)
                return dlog_prob * likelihood.compute_branch_length_derivatives(
                    q, category_rates, category_weights
                )

            log_prob_val = likelihood.compute_likelihood_from_partials(
                frequencies, category_weights
            )
            return log_prob_val, grad  # TODO: Cache site likelihoods

    else:

        def log_prob_1d(branch_lengths):
            eigendecomp = subst_model.eigen(frequencies, **subst_model_params)
            transition_probs = treeflow.substitution_model.transition_probs(
                eigendecomp, category_rates, branch_lengths
            )
            likelihood.compute_postorder_partials(transition_probs)
            log_prob_val = likelihood.compute_likelihood_from_partials(
                frequencies, category_weights
            )
            return log_prob_val

    def log_prob(branch_lengths):
        batch_shape = branch_lengths.shape[:-1]
        return treeflow.tf_util.vectorize_1d_if_needed(
            log_prob_1d, branch_lengths, batch_shape.rank
        )

    return log_prob, likelihood


class LeafSequences(tfp.distributions.Distribution):
    def __init__(
        self,
        tree,
        subst_model,
        frequencies,
        category_weights,
        category_rates,
        validate_args=False,
        allow_nan_stats=True,
        **subst_model_params
    ):
        super(LeafSequences, self).__init__(
            dtype={"sequences": tf.int64, "weights": tf.int64},
            reparameterization_type=tfp.distributions.NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=dict(locals()),
        )
        self.tree = tree
        self.subst_model = subst_model
        self.frequencies = frequencies
        self.category_weights = category_weights
        self.category_rates = category_rates
        self.subst_model_params = subst_model_params

    def _log_prob(self, value):
        return log_prob_conditioned(
            value, self.tree["topology"], len(self.category_weights)
        )(
            self.tree["heights"],
            self.subst_model,
            self.category_weights,
            self.category_rates,
            self.frequencies,
            **self.subst_model_params
        )

    def _sample_n(self, n, seed=None):
        raise NotImplementedError("Sequence simulator not yet implemented")
