import pytest
from numpy.testing import assert_allclose
import numpy as np
import tensorflow as tf
import libsbn
import treeflow.beagle
import treeflow.substitution_model
import treeflow.tensorflow_likelihood
import treeflow.sequences
import treeflow.tree_processing

# TF function: log_prob_conditioned
# libsbn function: log_prob_conditioned_branch_only
def test_log_prob_conditioned_hky(prep_likelihood, hky_params, weights_rates, hello_newick_file, hello_fasta_file):
    subst_model = treeflow.substitution_model.HKY()
    category_weights, category_rates = weights_rates
    tf_likelihood, tf_branch_lengths, tf_eigen = prep_likelihood(hello_newick_file, hello_fasta_file, subst_model, category_rates, category_weights, **hky_params)
    
    tree, taxon_names = treeflow.tree_processing.parse_newick(hello_newick_file)

    value = treeflow.sequences.get_encoded_sequences(hello_fasta_file, taxon_names)

    tf_log_prob = treeflow.sequences.log_prob_conditioned(value,tree['topology'],len(category_rates))

    libsbn_tf_func, inst = treeflow.beagle.log_prob_conditioned_branch_only(hello_fasta_file, subst_model,rescaling = False, inst = None, newick_file = hello_newick_file, **hky_params)
    
    libsbn_branch_lengths = np.array(inst.tree_collection.trees[0].branch_lengths)
    
    # Checking that branch lengths are equal. Note this fails at tolerance 1e-16
    assert_allclose(tf_branch_lengths, libsbn_branch_lengths[:-1], 1e-15)

    
    with tf.GradientTape() as tf_t:
        tf_t.watch(tf_branch_lengths)
        tf_ll = tf_log_prob(tf_branch_lengths, subst_model, category_weights, category_rates, **hky_params)
    tf_gradient = tf_t.gradient(tf_ll, tf_branch_lengths)
    
    
    with tf.GradientTape() as libsbn_t:
        libsbn_t.watch(tf_branch_lengths)
        libsbn_ll = libsbn_tf_func(tf_branch_lengths)
    libsbn_gradient = libsbn_t.gradient(libsbn_ll, tf_branch_lengths)
    
    assert_allclose(tf_ll.numpy(), libsbn_ll.numpy())
    assert_allclose(tf_gradient.numpy(), libsbn_gradient.numpy())


###############
# TF function: log_prob_conditioned_branch_only
# libsbn function: log_prob_condition_branch_only
'''
def test_log_prob_conditioned_branch_only_hky(prep_likelihood, hky_params, weights_rates, hello_newick_file, hello_fasta_file):
    subst_model = treeflow.substitution_model.HKY()
    category_weights, category_rates = weights_rates
    tf_branch_lengths, tf_eigen = prep_likelihood(hello_newick_file, hello_fasta_file, subst_model, category_rates, category_weights, **hky_params)[1:]
    
    tree, taxon_names = treeflow.tree_processing.parse_newick(hello_newick_file)

    value = treeflow.sequences.get_encoded_sequences(hello_fasta_file, taxon_names)

    tf_log_prob, tf_likelihood  = treeflow.sequences.log_prob_conditioned_branch_only(value,tree['topology'],len(category_rates),subst_model,category_weights,category_rates,**hky_params)

    libsbn_tf_func, inst = treeflow.beagle.log_prob_conditioned_branch_only(hello_fasta_file, subst_model,rescaling = False, inst = None, newick_file = hello_newick_file, **hky_params)
    
    libsbn_branch_lengths = np.array(inst.tree_collection.trees[0].branch_lengths)
    
    # Checking that branch lengths are equal. Note this fails at tolerance 1e-16
    assert_allclose(tf_branch_lengths, libsbn_branch_lengths[:-1], 1e-15)
    


    # treeflow likelihood
    # print(np.array(tf_likelihood.compute_likelihood_from_partials(hky_params['frequencies'],category_weights)))

    # libsbn likelihood
    #print(np.array(inst.phylo_gradients()[0].log_likelihood))
    #assert_allclose(np.array(tf_likelihood), np.array(inst.log_likelihoods()))
    

    with tf.GradientTape() as tf_t:
        tf_t.watch(tf_branch_lengths)
        tf_ll = tf_log_prob(tf_branch_lengths, subst_model, category_weights, category_rates, **hky_params)
    tf_gradient = tf_t.gradient(tf_ll, tf_branch_lengths)
    
    
    with tf.GradientTape() as libsbn_t:
        libsbn_t.watch(tf_branch_lengths)
        libsbn_ll = libsbn_tf_func(tf_branch_lengths)
    libsbn_gradient = libsbn_t.gradient(libsbn_ll, tf_branch_lengths)
    assert_allclose(tf_gradient.numpy(), libsbn_gradient.numpy())
'''
