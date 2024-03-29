import pytest
from numpy.testing import assert_allclose
import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.evolution.substitution.nucleotide.hky import HKY
from treeflow.evolution.substitution.probabilities import (
    get_transition_probabilities_eigen,
)
from treeflow.evolution.seqio import Alignment
from treeflow.traversal.phylo_likelihood import phylogenetic_likelihood
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree


@pytest.mark.parametrize("function_mode", [True, False])
def test_phylo_likelihood_hky_beast(
    hello_tensor_tree: TensorflowRootedTree,
    hello_alignment: Alignment,
    function_mode: bool,
    hky_params,
    hello_hky_log_likelihood: float,
):
    subst_model = HKY()
    eigen = subst_model.eigen(**hky_params)
    probs = tf.expand_dims(
        get_transition_probabilities_eigen(eigen, hello_tensor_tree.branch_lengths), 0
    )
    encoded_sequences = hello_alignment.get_encoded_sequence_tensor(
        hello_tensor_tree.taxon_set
    )
    if function_mode:
        func = tf.function(phylogenetic_likelihood)
    else:
        func = phylogenetic_likelihood
    site_partials = func(
        encoded_sequences,
        probs,
        hky_params["frequencies"],
        hello_tensor_tree.topology.postorder_node_indices,
        hello_tensor_tree.topology.node_child_indices,
        batch_shape=tf.shape(encoded_sequences)[:1],
    )
    res = tf.reduce_sum(tf.math.log(site_partials))
    expected = hello_hky_log_likelihood
    assert_allclose(res.numpy(), expected)
