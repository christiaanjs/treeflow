import pytest
from treeflow.tree.io import parse_newick
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from treeflow.evolution.seqio import Alignment


@pytest.fixture
def hello_newick_file(test_data_dir):
    return str(test_data_dir / "hello.nwk")


@pytest.fixture
def hello_fasta_file(test_data_dir):
    return str(test_data_dir / "hello.fasta")


@pytest.fixture
def hello_tensor_tree(hello_newick_file):
    numpy_tree = parse_newick(hello_newick_file)
    return convert_tree_to_tensor(numpy_tree)


@pytest.fixture
def hello_alignment(hello_fasta_file):
    return Alignment(hello_fasta_file)
