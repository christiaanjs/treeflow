import pytest
from numpy.testing import assert_allclose
import tensorflow as tf
from treeflow.evolution.seqio import Alignment
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from treeflow.evolution.substitution.nucleotide.hky import HKY
from treeflow.evolution.substitution.probabilities import (
    get_transition_probabilities_eigen,
)
from treeflow.traversal.postorder import postorder_node_traversal
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.traversal.phylo_likelihood import move_indices_to_outside
from treeflow.tree.topology.numpy_tree_topology import StaticNumpyTreeTopology


def traversal_likelihood(
    encoded_sequences,
    transition_probabilities,
    frequencies,
    topology: TensorflowTreeTopology,
    batch_rank=1,
    unroll: bool = False,
    xla_compatible: bool = False,
):
    child_transition_probs_child_first = tf.gather(
        transition_probabilities, topology.node_child_indices, axis=-3
    )
    child_transition_probs = move_indices_to_outside(
        child_transition_probs_child_first,
        batch_rank,
        2,
    )
    leaf_init = move_indices_to_outside(encoded_sequences, batch_rank, 1)

    def mapping(child_partials, node_child_transition_probs, topology_info):
        parent_child_probs = node_child_transition_probs * tf.expand_dims(
            child_partials, -2
        )
        return tf.reduce_prod(
            tf.reduce_sum(
                parent_child_probs,
                axis=-1,
            ),
            axis=0,
        )

    partials = postorder_node_traversal(
        topology,
        mapping,
        child_transition_probs,
        leaf_init,
        unroll=unroll,
        xla_compatible=xla_compatible,
    )
    root_partials = partials[-1]
    return tf.reduce_sum(frequencies * root_partials, axis=-1)


@pytest.mark.parametrize("xla_compatible", [True, False])
@pytest.mark.parametrize("function_mode", [True, False])
@pytest.mark.parametrize("unroll", [True, False])
def test_postorder_node_traversal_phylo_likelihood(
    hello_tensor_tree: TensorflowRootedTree,
    hello_alignment: Alignment,
    function_mode: bool,
    unroll: bool,
    xla_compatible: bool,
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
        func = tf.function(traversal_likelihood, jit_compile=xla_compatible)
    else:
        func = traversal_likelihood

    if unroll and function_mode:
        topology = StaticNumpyTreeTopology.from_numpy_topology(
            hello_tensor_tree.topology.numpy()
        )
    else:
        topology = hello_tensor_tree.topology

    site_partials = func(
        encoded_sequences,
        probs,
        hky_params["frequencies"],
        topology,
        unroll=unroll,
        xla_compatible=xla_compatible,
    )
    res = tf.reduce_sum(tf.math.log(site_partials))
    expected = hello_hky_log_likelihood
    assert_allclose(res.numpy(), expected)
