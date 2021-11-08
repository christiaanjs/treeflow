import pytest
import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.tree.io import parse_newick
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from treeflow.evolution.substitution.nucleotide.hky import HKY
from treeflow.evolution.seqio import Alignment
from treeflow.traversal.phylo_likelihood import phylogenetic_likelihood
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree


@pytest.fixture
def hello_tensor_tree(hello_newick_file):
    numpy_tree = parse_newick(hello_newick_file)
    return convert_tree_to_tensor(numpy_tree)


@pytest.fixture
def hello_alignment(hello_newick_file):
    return Alignment(hello_newick_file)


@pytest.mark.parametrize("function_mode", [True, False])
def test_phylo_likelihood_hky_beast(
    hello_tensor_tree: TensorflowRootedTree,
    hello_alignment: Alignment,
    function_mode: bool,
):
    subst_model = HKY()
    assert False
