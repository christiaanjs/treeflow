import pytest
from treeflow.tree.io import parse_newick
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from treeflow.evolution.seqio import Alignment

_HELLO_NEWICK = "hello.nwk"


@pytest.fixture
def hello_newick_file(test_data_dir):
    return str(test_data_dir / _HELLO_NEWICK)


_HELLO_FASTA = "hello.fasta"


@pytest.fixture
def hello_fasta_file(test_data_dir):
    return str(test_data_dir / _HELLO_FASTA)


@pytest.fixture
def hello_tensor_tree(hello_newick_file):
    numpy_tree = parse_newick(hello_newick_file)
    return convert_tree_to_tensor(numpy_tree)


@pytest.fixture
def hello_alignment(hello_fasta_file):
    return Alignment(hello_fasta_file)


_WNV_NEWICK = "wnv.nwk"


@pytest.fixture
def wnv_newick_file(test_data_dir):
    return str(test_data_dir / _WNV_NEWICK)


_WNV_FASTA = "wnv.fasta"


@pytest.fixture
def wnv_fasta_file(test_data_dir):
    return str(test_data_dir / _WNV_FASTA)


@pytest.fixture(
    params=[
        (
            _HELLO_NEWICK,
            _HELLO_FASTA,
            False,
        ),
        (
            _WNV_NEWICK,
            _WNV_FASTA,
            True,
        ),
    ]
)
def newick_fasta_file_dated(test_data_dir, request):
    newick_file, fasta_file, dated = request.param
    return str(test_data_dir / newick_file), str(test_data_dir / fasta_file), dated


@pytest.fixture
def newick_file_dated(newick_fasta_file_dated):
    newick_file, _, dated = newick_fasta_file_dated
    return newick_file, dated


@pytest.fixture
def tree_sim_newick_file(test_data_dir):
    return str(test_data_dir / "tree-sim.newick")
