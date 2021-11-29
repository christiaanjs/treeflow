from tensorflow_probability.python.distributions.sample import Sample
from treeflow.distributions.sample_weighted import SampleWeighted
from treeflow.distributions.leaf_ctmc import LeafCTMC
from treeflow.tree.io import parse_newick
from treeflow.evolution.seqio import parse_fasta, Alignment
from treeflow.tree.rooted.tensorflow_rooted_tree import (
    convert_tree_to_tensor,
    TensorflowRootedTree,
)
from treeflow.tree.unrooted.tensorflow_unrooted_tree import TensorflowUnrootedTree
from treeflow.evolution.substitution.nucleotide.hky import HKY
from treeflow.evolution.substitution.probabilities import (
    get_transition_probabilities_tree,
)
from numpy.testing import assert_allclose


def compute_log_prob_uncompressed(
    alignment: Alignment,
    transition_probs_tree: TensorflowUnrootedTree,
    frequencies,
):
    sequences_encoded = alignment.get_encoded_sequence_tensor(
        transition_probs_tree.taxon_set
    )
    dist = Sample(
        LeafCTMC(transition_probs_tree, frequencies),
        sample_shape=(alignment.site_count,),
    )
    return dist.log_prob(sequences_encoded)


def compute_log_prob_compressed(
    alignment: Alignment,
    transition_probs_tree: TensorflowUnrootedTree,
    frequencies,
):
    weighted_alignment = alignment.get_compressed_alignment()
    sequences_encoded = weighted_alignment.get_encoded_sequence_tensor(
        transition_probs_tree.taxon_set
    )
    dist = SampleWeighted(
        LeafCTMC(transition_probs_tree, frequencies),
        weights=weighted_alignment.get_weights_tensor(),
        sample_shape=(weighted_alignment.site_count,),
    )
    return dist.log_prob(sequences_encoded)


def test_sample_weighted(wnv_newick_file, wnv_fasta_file, hky_params):
    alignment = Alignment(wnv_fasta_file)
    numpy_tree = parse_newick(wnv_newick_file)
    tensor_tree = convert_tree_to_tensor(numpy_tree)
    transition_prob_tree = get_transition_probabilities_tree(
        tensor_tree.get_unrooted_tree(), HKY(), **hky_params
    )

    log_prob_uncompressed = compute_log_prob_uncompressed(
        alignment, transition_prob_tree, hky_params["frequencies"]
    )
    log_prob_compressed = compute_log_prob_compressed(
        alignment, transition_prob_tree, hky_params["frequencies"]
    )
    assert_allclose(log_prob_uncompressed.numpy(), log_prob_compressed.numpy())
