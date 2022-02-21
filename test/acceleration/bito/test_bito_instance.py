import pytest

from treeflow.tree.io import parse_newick
from treeflow.traversal.anchor_heights import get_anchor_heights
from numpy.testing import assert_allclose, assert_equal


# TODO: Test taxon name order

def test_get_tree_info(newick_file_dated):
    from treeflow.acceleration.bito.instance import get_instance, get_tree_info

    newick_file, dated = newick_file_dated
    treeflow_tree = parse_newick(newick_file)
    treeflow_node_bounds = get_anchor_heights(treeflow_tree)

    inst = get_instance(newick_file, dated=dated)
    bito_tree, bito_node_bounds = get_tree_info(inst)

    assert_equal(
        treeflow_tree.topology.parent_indices,
        bito_tree.topology.parent_indices,
    )
    assert_allclose(treeflow_tree.heights, bito_tree.heights, atol=1e-12)
    assert_allclose(treeflow_node_bounds, bito_node_bounds, atol=1e-12)
