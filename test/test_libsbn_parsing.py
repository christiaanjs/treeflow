import pytest
import treeflow.tree_processing
import treeflow.libsbn
from numpy.testing import assert_allclose, assert_equal


def get_tree_infos(newick_file_dated):
    newick_file, dated = newick_file_dated
    tf_tree_info = treeflow.tree_processing.parse_tree_info(newick_file)

    inst = treeflow.libsbn.get_instance(newick_file, dated=dated)
    libsbn_tree_info = treeflow.libsbn.get_tree_info(inst)
    return inst, tf_tree_info, libsbn_tree_info


# TODO: Test taxon name order


def test_parent_topology_parsing(newick_file_dated):
    _, tf_tree_info, libsbn_tree_info = get_tree_infos(newick_file_dated)
    assert_equal(
        tf_tree_info.tree["topology"]["parent_indices"],
        libsbn_tree_info.tree["topology"]["parent_indices"],
    )


def test_height_parsing(newick_file_dated):
    _, tf_tree_info, libsbn_tree_info = get_tree_infos(newick_file_dated)
    assert_allclose(
        tf_tree_info.tree["heights"], libsbn_tree_info.tree["heights"], atol=1e-12
    )


def test_node_bound_parsing(newick_file_dated):
    _, tf_tree_info, libsbn_tree_info = get_tree_infos(newick_file_dated)
    assert_allclose(tf_tree_info.node_bounds, libsbn_tree_info.node_bounds, atol=1e-12)
