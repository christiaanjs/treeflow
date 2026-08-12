import numpy as np
import pytest

from treeflow.vbpi.clade import (
    canonical_subsplit,
    decompose_topology,
    node_clades,
    root_clade,
)


def test_node_clades_balanced():
    # balanced ((0,1),(2,3)): 0,1->4 ; 2,3->5 ; 4,5->6
    parent_indices = np.array([4, 4, 5, 5, 6, 6])
    clades = node_clades(parent_indices, taxon_count=4)
    assert clades[0] == frozenset({0})
    assert clades[4] == frozenset({0, 1})
    assert clades[5] == frozenset({2, 3})
    assert clades[6] == frozenset({0, 1, 2, 3})


def test_decompose_matches_internal_nodes():
    parent_indices = np.array([4, 4, 5, 5, 6, 6])
    pairs = decompose_topology(parent_indices, taxon_count=4)
    # one factor per internal node (n - 1 = 3)
    assert len(pairs) == 3
    parents = {p for p, _ in pairs}
    assert root_clade(4) in parents
    root_subsplit = dict(pairs)[root_clade(4)]
    assert set(root_subsplit) == {frozenset({0, 1}), frozenset({2, 3})}


def test_canonical_subsplit_is_order_independent():
    a = frozenset({0, 1})
    b = frozenset({2})
    assert canonical_subsplit(a, b) == canonical_subsplit(b, a)
    # smaller (by size then taxa) comes first
    left, right = canonical_subsplit(a, b)
    assert left == frozenset({2})


def test_decompose_rejects_wrong_length():
    with pytest.raises(ValueError):
        node_clades(np.array([4, 4, 5]), taxon_count=4)


def test_caterpillar_topology_clades():
    # caterpillar (((0,1),2),3): 0,1->4 ; 4,2->5 ; 5,3->6
    # parent_indices indexed by node id 0..5:
    #   node0->4 node1->4 node2->5 node3->6 node4->5 node5->6
    ladder = np.array([4, 4, 5, 6, 5, 6])
    clades = node_clades(ladder, taxon_count=4)
    assert clades[4] == frozenset({0, 1})
    assert clades[5] == frozenset({0, 1, 2})
    assert clades[6] == root_clade(4)
    pairs = dict(decompose_topology(ladder, taxon_count=4))
    # the node-5 clade {0,1,2} splits into {0,1} and {2}
    assert set(pairs[frozenset({0, 1, 2})]) == {frozenset({0, 1}), frozenset({2})}
