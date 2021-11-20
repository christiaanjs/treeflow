import pytest
from numpy.testing import assert_allclose
import tensorflow as tf
from treeflow.evolution.seqio import Alignment
from treeflow.evolution.substitution.nucleotide.hky import HKY
from treeflow.evolution.substitution.probabilities import (
    get_transition_probabilities_tree,
)
from treeflow.tree.io import parse_newick
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from tensorflow_probability.python.distributions import Sample
from treeflow.distributions.leaf_ctmc import LeafCTMC


@pytest.mark.skip
def test_log_prob_conditioned_hky(hky_params, newick_fasta_file_dated):
    from treeflow.acceleration.bito.beagle import (
        phylogenetic_likelihood as beagle_likelihood,
    )

    newick_file, fasta_file, dated = newick_fasta_file_dated
    subst_model = HKY()
    tensor_tree = convert_tree_to_tensor(parse_newick(newick_file)).get_unrooted_tree()
    alignment = Alignment(fasta_file)
    sequences = alignment.get_encoded_sequence_tensor(tensor_tree.taxon_set)
    treeflow_func = lambda blens: Sample(
        LeafCTMC(
            get_transition_probabilities_tree(
                tensor_tree.with_branch_lengths(blens), subst_model, **hky_params
            ),
            hky_params["frequencies"],
        ),
        sample_shape=alignment.site_count,
    ).log_prob(sequences)

    beagle_func, _ = beagle_likelihood(
        fasta_file, subst_model, newick_file=newick_file, dated=dated, **hky_params
    )

    blens = tensor_tree.branch_lengths
    with tf.GradientTape() as tf_t:
        tf_t.watch(blens)
        tf_ll = treeflow_func(blens)
    tf_gradient = tf_t.gradient(tf_ll, blens)

    with tf.GradientTape() as libsbn_t:
        libsbn_t.watch(blens)
        libsbn_ll = beagle_func(blens)
    libsbn_gradient = libsbn_t.gradient(libsbn_ll, blens)

    assert_allclose(tf_ll.numpy(), libsbn_ll.numpy())
    assert_allclose(tf_gradient.numpy(), libsbn_gradient.numpy())
