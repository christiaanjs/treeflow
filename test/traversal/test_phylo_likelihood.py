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
from treeflow.tree.topology.numpy_tree_topology import StaticNumpyTreeTopology


@pytest.mark.parametrize("function_mode", [True, False])
@pytest.mark.parametrize("unroll", ["unrolled", "tensorarray", "while_loop"])
def test_phylo_likelihood_hky_beast(
    hello_tensor_tree: TensorflowRootedTree,
    hello_alignment: Alignment,
    function_mode: bool,
    unroll: str,
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

    if unroll == "unrolled" and function_mode:
        # 'unrolled' needs the topology index values statically foldable inside the
        # traced function; the static NumPy topology guarantees that.
        topology = StaticNumpyTreeTopology.from_numpy_topology(
            hello_tensor_tree.topology.numpy()
        )
    else:
        topology = hello_tensor_tree.topology

    if function_mode:
        func = tf.function(phylogenetic_likelihood)
    else:
        func = phylogenetic_likelihood
    site_partials = func(
        topology,
        encoded_sequences,
        probs,
        hky_params["frequencies"],
        batch_shape=tf.shape(encoded_sequences)[:1],
        unroll=unroll,
    )
    res = tf.reduce_sum(tf.math.log(site_partials))
    expected = hello_hky_log_likelihood
    assert_allclose(res.numpy(), expected)
