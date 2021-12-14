from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.tree.taxon_set import TupleTaxonSet
import pytest
from pathlib import Path
from treeflow.tree.io import parse_newick, remove_zero_edges
import numpy as np
from numpy.testing import assert_allclose

from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology

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


def test_remove_zero_edges_no_changes(hello_newick_file: str):
    tree = parse_newick(hello_newick_file)
    starting_heights = np.array(tree.heights)
    res = remove_zero_edges(tree)
    assert res is not tree
    assert_allclose(tree.heights, starting_heights)
    assert_allclose(res.heights, tree.heights)  # No zero branches


def test_remove_zero_edges():
    node_height = 0.5
    epsilon = 0.01
    parent_indices = np.array([4, 4, 5, 6, 5, 6])
    heights = np.array([0.0, 0.1, 0.4, 0.2, node_height, node_height, node_height])
    expected_heights = np.concatenate(
        [heights[:4], [node_height, node_height + epsilon, node_height + 2 * epsilon]]
    )
    tree = NumpyRootedTree(heights, topology=NumpyTreeTopology(parent_indices))
    res = remove_zero_edges(tree, epsilon=epsilon)
    assert_allclose(res.heights, expected_heights)
