import pytest


@pytest.fixture
def hello_newick_file(test_data_dir):
    return str(test_data_dir / "hello.nwk")


@pytest.fixture
def hello_fasta_file(test_data_dir):
    return str(test_data_dir / "hello.fasta")
