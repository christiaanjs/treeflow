import pytest
from numpy.testing import assert_allclose
import numpy as np
import tensorflow as tf
import treeflow.beagle
import treeflow.substitution_model
import treeflow.tensorflow_likelihood
import treeflow.sequences
import treeflow.tree_processing

# TODO: Test Weibull site model
# TF function: log_prob_conditioned
# libsbn function: log_prob_conditioned_branch_only
def test_log_prob_conditioned_hky(
    prep_likelihood, hky_params, single_weights, single_rates, newick_fasta_file_dated
):
    newick_file, fasta_file, dated = newick_fasta_file_dated
    subst_model = treeflow.substitution_model.HKY()
    category_weights = single_weights
    category_rates = single_weights
    tf_likelihood, tf_branch_lengths, tf_eigen = prep_likelihood(
        newick_file,
        fasta_file,
        subst_model,
        category_rates,
        category_weights,
        **hky_params
    )

    tree, taxon_names = treeflow.tree_processing.parse_newick(newick_file)

    value = treeflow.sequences.get_encoded_sequences(fasta_file, taxon_names)

    tf_log_prob = treeflow.sequences.log_prob_conditioned(
        value, tree["topology"], len(category_rates)
    )

    libsbn_tf_func, inst = treeflow.beagle.log_prob_conditioned_branch_only(
        fasta_file, subst_model, newick_file=newick_file, dated=dated, **hky_params
    )

    with tf.GradientTape() as tf_t:
        tf_t.watch(tf_branch_lengths)
        tf_ll = tf_log_prob(
            tf_branch_lengths,
            subst_model,
            category_weights,
            category_rates,
            **hky_params
        )
    tf_gradient = tf_t.gradient(tf_ll, tf_branch_lengths)

    with tf.GradientTape() as libsbn_t:
        libsbn_t.watch(tf_branch_lengths)
        libsbn_ll = libsbn_tf_func(tf_branch_lengths)
    libsbn_gradient = libsbn_t.gradient(libsbn_ll, tf_branch_lengths)

    assert_allclose(tf_ll.numpy(), libsbn_ll.numpy())
    assert_allclose(tf_gradient.numpy(), libsbn_gradient.numpy())
