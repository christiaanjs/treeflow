import numpy as np
import pytest

from treeflow.conditional_clade.clade import clade_taxa, full_clade
from treeflow.conditional_clade.support import ConditionalCladeSupport
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_topology_count_matches_enumeration(n):
    support = ConditionalCladeSupport(n)
    enumerated = list(support.enumerate_assignments())
    assert len(enumerated) == support.topology_count()


@pytest.mark.parametrize("n", [3, 4, 5])
def test_assignment_parent_indices_roundtrip(n):
    support = ConditionalCladeSupport(n)
    for assignment in support.enumerate_assignments():
        parent_indices = support.assignment_to_parent_indices(assignment)
        recovered = support.parent_indices_to_assignment(parent_indices)
        assert recovered == assignment


@pytest.mark.parametrize("n", [3, 4, 5])
def test_parent_indices_are_valid_topologies(n):
    support = ConditionalCladeSupport(n)
    node_count = 2 * n - 1
    for parent_indices in support.enumerate_parent_indices():
        assert parent_indices.shape == (node_count - 1,)
        # leaves 0..n-1, internal n..2n-2, root last; parent index strictly larger
        for i, parent in enumerate(parent_indices):
            assert parent > i
            assert n <= parent <= node_count - 1
        # the derived child structure must be bifurcating and well formed
        topology = NumpyTreeTopology(parent_indices=parent_indices)
        child_indices = topology.child_indices
        for node in range(n, node_count):
            assert np.all(child_indices[node] >= 0)
        # leaves have no children
        for leaf in range(n):
            assert np.all(child_indices[leaf] == -1)


def test_enumerated_topologies_distinct():
    support = ConditionalCladeSupport(5)
    seen = {tuple(pi.tolist()) for pi in support.enumerate_parent_indices()}
    assert len(seen) == support.topology_count()


def test_flat_indexing_consistency():
    support = ConditionalCladeSupport(4)
    assert support.subsplit_count == len(support.flat_subsplits)
    assert support.segment_ids.shape == (support.subsplit_count,)
    # each (parent, subsplit) maps to a unique flat index covering 0..M-1
    indices = sorted(support.flat_index.values())
    assert indices == list(range(support.subsplit_count))
    # segment id of a flat subsplit points back to its parent clade
    for (parent, _subsplit), flat in support.flat_index.items():
        assert support.parent_clades[support.segment_ids[flat]] == parent


def test_assignment_flat_indices_length():
    n = 5
    support = ConditionalCladeSupport(n)
    for assignment in support.enumerate_assignments():
        flat = support.assignment_flat_indices(assignment)
        # every rooted tree on n taxa has exactly n-1 internal nodes
        assert flat.shape == (n - 1,)
        assert len(set(flat.tolist())) == n - 1


def test_feature_matrices_shapes():
    n = 4
    support = ConditionalCladeSupport(n)
    clade_features = support.clade_feature_matrix()
    assert clade_features.shape == (support.parent_clade_count, n)
    subsplit_features = support.subsplit_feature_matrix()
    assert subsplit_features.shape == (support.subsplit_count, 3 * n)
    # parent feature block equals union of the two child blocks
    parent_block = subsplit_features[:, :n]
    child1_block = subsplit_features[:, n : 2 * n]
    child2_block = subsplit_features[:, 2 * n :]
    np.testing.assert_array_equal(parent_block, child1_block + child2_block)


def test_known_three_taxon_topologies():
    support = ConditionalCladeSupport(3)
    # three rooted topologies on 3 taxa
    assert support.topology_count() == 3
    parents = {full_clade(3)}
    # root clade always present, and three possible cherries
    cherries = set()
    for assignment in support.enumerate_assignments():
        root_subsplit = assignment[full_clade(3)]
        cherry = max(
            (root_subsplit.child1, root_subsplit.child2), key=lambda c: bin(c).count("1")
        )
        cherries.add(clade_taxa(cherry))
    assert cherries == {(0, 1), (0, 2), (1, 2)}
