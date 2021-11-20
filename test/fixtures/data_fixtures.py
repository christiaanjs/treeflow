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


@pytest.fixture
def wnv_newick_file(test_data_dir):
    return str(test_data_dir / "wnv.nwk")


@pytest.fixture
def wnv_fasta_file(test_data_dir):
    return str(test_data_dir / "wnv.fasta")


@pytest.fixture(
    params=[
        (
            hello_newick_file._pytestfixturefunction,
            hello_fasta_file._pytestfixturefunction,
            False,
        ),
        (
            wnv_newick_file._pytestfixturefunction,
            wnv_fasta_file._pytestfixturefunction,
            True,
        ),
    ]
)
def newick_fasta_file_dated(test_data_dir, request):
    newick_file_func, fasta_file_func, dated = request.param
    return newick_file_func(test_data_dir), fasta_file_func(test_data_dir), dated


@pytest.fixture
def newick_file_dated(newick_fasta_file_dated):
    newick_file, _, dated = newick_fasta_file_dated
    return newick_file, dated
