import pytest

from treeflow.conditional_clade.clade import (
    Subsplit,
    binary_vector_to_clade,
    clade_size,
    clade_taxa,
    clade_to_binary_vector,
    enumerate_clade_subsplits,
    full_clade,
    is_singleton,
    make_subsplit,
    min_taxon,
    taxa_to_clade,
)


def test_taxa_clade_roundtrip():
    taxa = (0, 2, 5)
    clade = taxa_to_clade(taxa)
    assert clade_taxa(clade) == taxa
    assert clade_size(clade) == 3
    assert min_taxon(clade) == 0


def test_binary_vector_roundtrip():
    clade = taxa_to_clade((1, 3))
    vector = clade_to_binary_vector(clade, 5)
    assert vector == [0, 1, 0, 1, 0]
    assert binary_vector_to_clade(vector) == clade


def test_full_clade_and_singleton():
    assert full_clade(4) == 0b1111
    assert is_singleton(taxa_to_clade((2,)))
    assert not is_singleton(taxa_to_clade((1, 2)))


def test_make_subsplit_canonical():
    a = taxa_to_clade((2, 3))
    b = taxa_to_clade((0, 1))
    subsplit = make_subsplit(a, b)
    # child1 must contain the smallest taxon of the union
    assert min_taxon(subsplit.child1) == 0
    assert subsplit.child1 == b
    assert subsplit.parent == taxa_to_clade((0, 1, 2, 3))


def test_make_subsplit_validates():
    with pytest.raises(ValueError):
        make_subsplit(taxa_to_clade((0, 1)), taxa_to_clade((1, 2)))  # overlap
    with pytest.raises(ValueError):
        make_subsplit(0, taxa_to_clade((1,)))  # empty


@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_enumerate_subsplit_count(size):
    clade = full_clade(size)
    subsplits = enumerate_clade_subsplits(clade)
    assert len(subsplits) == 2 ** (size - 1) - 1
    # all canonical, disjoint, covering the clade, unique
    seen = set()
    smallest = min_taxon(clade)
    for s in subsplits:
        assert isinstance(s, Subsplit)
        assert s.child1 & s.child2 == 0
        assert s.parent == clade
        assert min_taxon(s.child1) == smallest
        seen.add(s)
    assert len(seen) == len(subsplits)


def test_singleton_has_no_subsplits():
    assert enumerate_clade_subsplits(taxa_to_clade((3,))) == []
