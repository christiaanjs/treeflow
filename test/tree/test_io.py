import typing as tp
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import assert_allclose
import tensorflow as tf

from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from treeflow.tree.taxon_set import TupleTaxonSet
from treeflow.tree.io import parse_newick, remove_zero_edges, write_tensor_trees
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


@pytest.mark.parametrize("branch_metadata_keys", [None, ["foo"]])
def test_write_tensor_trees(
    hello_newick_file: str,
    hello_tensor_tree: TensorflowRootedTree,
    tmp_path: Path,
    branch_metadata_keys: tp.Optional[tp.List[str]],
):
    vec_branch_lengths = (
        tf.expand_dims(
            tf.range(1, 5, dtype=hello_tensor_tree.branch_lengths.dtype) / 7.0, -1
        )
        * hello_tensor_tree.branch_lengths
    )
    if branch_metadata_keys is None:
        branch_metadata = {}
    else:
        branch_metadata = {
            key: vec_branch_lengths * float(i + 2)
            for i, key in enumerate(branch_metadata_keys)
        }
        print(branch_metadata)
    output_file = tmp_path / "foo.nexus"
    write_tensor_trees(
        hello_newick_file,
        vec_branch_lengths,
        output_file,
        branch_metadata=branch_metadata,
    )

    with open(output_file) as f:
        print(f.read())
