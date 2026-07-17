import numpy as np
import pytest

from treeflow.vbpi.support import SubsplitSupport, all_rooted_parent_indices


def double_factorial_odd(k):
    """(2k-3)!! = number of rooted binary topologies on k taxa."""
    result = 1
    for m in range(1, 2 * k - 2, 2):
        result *= m
    return result


@pytest.mark.parametrize("taxon_count", [2, 3, 4, 5, 6])
def test_enumeration_counts(taxon_count):
    topos = all_rooted_parent_indices(taxon_count)
    assert len(topos) == double_factorial_odd(taxon_count)
    # all distinct
    keys = {tuple(t.tolist()) for t in topos}
    assert len(keys) == len(topos)


def test_enumerated_topologies_are_valid():
    for t in all_rooted_parent_indices(5):
        # leaves 0..4, internal 5..8, root 8, children < parent
        assert t.shape == (8,)
        for node, parent in enumerate(t):
            assert parent > node  # children have smaller ids than parents
        assert t.max() == 8  # root id is 2n-2


def test_csr_offsets_consistent():
    support = SubsplitSupport.from_topologies(all_rooted_parent_indices(5), 5)
    assert support.child_offsets.shape == (support.num_clades + 1,)
    assert support.child_offsets[0] == 0
    assert support.child_offsets[-1] == support.num_candidates
    # offsets non-decreasing
    assert np.all(np.diff(support.child_offsets) >= 0)
    # candidate_parent_clade must equal the CSR segment each candidate lies in
    for c in range(support.num_clades):
        start, end = support.child_offsets[c], support.child_offsets[c + 1]
        assert np.all(support.candidate_parent_clade[start:end] == c)
    # segment ids therefore non-decreasing (required by tf.math.segment_*)
    assert np.all(np.diff(support.candidate_parent_clade) >= 0)


def test_leaf_clades_have_no_candidates():
    support = SubsplitSupport.from_topologies(all_rooted_parent_indices(5), 5)
    for c in range(support.num_clades):
        if support.clade_leaf_taxon[c] >= 0:
            assert support.child_offsets[c] == support.child_offsets[c + 1]


def test_candidate_indices_roundtrip():
    topos = all_rooted_parent_indices(5)
    support = SubsplitSupport.from_topologies(topos, 5)
    for t in topos:
        idx = support.topology_candidate_indices(t)
        assert idx.shape == (4,)  # n - 1 internal nodes
        assert support.contains_topology(t)


def test_out_of_support_topology_detected():
    # Support from a single topology; a different one is out of support.
    topos = all_rooted_parent_indices(5)
    support = SubsplitSupport.from_topologies([topos[0]], 5)
    assert support.contains_topology(topos[0])
    # find a topology not representable
    missing = [t for t in topos if not support.contains_topology(t)]
    assert missing  # there must be some
    with pytest.raises(KeyError):
        support.topology_candidate_indices(missing[0])


def test_complete_support_contains_every_topology(make_random_topologies):
    topos = all_rooted_parent_indices(6)
    support = SubsplitSupport.from_topologies(topos, 6)
    for t in make_random_topologies(6, 30, seed=3):
        assert support.contains_topology(t)
