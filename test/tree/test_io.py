from treeflow.tree.taxon_set import TupleTaxonSet
import pytest
from pathlib import Path
from treeflow.tree.io import parse_newick
import numpy as np
from numpy.testing import assert_allclose

data_dir = Path("test/data")


@pytest.fixture
def hello_newick_file():
    return str(data_dir / "hello.nwk")


def test_parse_newick(hello_newick_file: str):
    tree = parse_newick(hello_newick_file)
    expected_heights = np.array([0.0, 0.0, 0.0, 0.1, 0.3])
    expected_parent_indices = np.array([3, 3, 4, 4])
    expected_taxon_names = ["mars", "saturn", "jupiter"]

    assert tree.taxon_set == TupleTaxonSet(expected_taxon_names)
    assert_allclose(tree.heights, expected_heights, atol=1e-16)
    assert_allclose(tree.topology.parent_indices, expected_parent_indices)
