"""Core representations for clades and subsplits.

A *clade* is a non-empty subset of the taxa. In a rooted tree it is the set of
leaves descended from some node. We represent a clade canonically as a Python
integer *bitset* over the (ordered) taxon set: bit ``i`` is set iff taxon ``i``
is a member. Integers are hashable, cheap to combine (union is ``|``,
intersection ``&``) and make excellent dictionary keys, which is convenient for
the dynamic programs over clades used elsewhere in this package.

A *subsplit* describes how a (parent) clade is divided into its two child
clades by an internal node of a rooted tree. It is the atomic event of a
conditional clade / subsplit Bayesian network (SBN) distribution over
topologies: the probability of a tree factorises into a product of conditional
probabilities ``P(subsplit | parent clade)`` over the internal nodes.

The two children of a subsplit are unordered, so we impose a canonical ordering
to give each subsplit a unique representation: ``child1`` is always the child
that contains the smallest-indexed taxon of the parent clade.

Binary-vector views
-------------------
Every clade can be viewed as a length-``n`` binary vector
(:func:`clade_to_binary_vector`). This is deliberate: it lets a parametrisation
of the conditional probabilities be driven by an *embedding* of the clade
(or subsplit) binary vectors -- e.g. a small neural network mapping
``(parent, child1, child2)`` binary vectors to a logit -- rather than storing an
explicit probability for every subsplit. The enumeration machinery here still
materialises every subsplit (for exact, small-taxon-set computations), but
nothing about the representation forces that: see
:meth:`treeflow.conditional_clade.support.ConditionalCladeSupport.subsplit_feature_matrix`.
"""

from __future__ import annotations

import typing as tp


class Subsplit(tp.NamedTuple):
    """An ordered pair of disjoint child clades produced by splitting a parent.

    ``child1`` and ``child2`` are bitsets. By convention (enforced by
    :func:`make_subsplit`) ``child1`` contains the smallest-indexed taxon of the
    parent clade ``child1 | child2``. The two children are disjoint and
    non-empty.
    """

    child1: int
    child2: int

    @property
    def parent(self) -> int:
        """The (bitset of the) parent clade ``child1 | child2``."""
        return self.child1 | self.child2


def popcount(clade: int) -> int:
    """Number of taxa in ``clade`` (the population count of the bitset)."""
    return bin(clade).count("1")


# Alias with domain meaning.
clade_size = popcount


def is_singleton(clade: int) -> bool:
    """Whether ``clade`` contains exactly one taxon (i.e. is a leaf clade)."""
    return clade != 0 and (clade & (clade - 1)) == 0


def min_taxon(clade: int) -> int:
    """Index of the smallest-indexed taxon in ``clade``.

    Raises ``ValueError`` for the empty clade.
    """
    if clade == 0:
        raise ValueError("Empty clade has no taxa")
    return (clade & -clade).bit_length() - 1


def clade_taxa(clade: int) -> tp.Tuple[int, ...]:
    """The taxon indices in ``clade`` as a sorted tuple."""
    taxa = []
    remaining = clade
    while remaining:
        low = remaining & -remaining
        taxa.append(low.bit_length() - 1)
        remaining ^= low
    return tuple(taxa)


def taxa_to_clade(taxa: tp.Iterable[int]) -> int:
    """Bitset for the given taxon indices."""
    clade = 0
    for taxon in taxa:
        clade |= 1 << taxon
    return clade


def full_clade(taxon_count: int) -> int:
    """The clade containing every taxon (the root clade)."""
    return (1 << taxon_count) - 1


def clade_to_binary_vector(clade: int, taxon_count: int) -> tp.List[int]:
    """Length-``taxon_count`` 0/1 membership vector for ``clade``.

    Returned as a plain list so this module stays dependency-free; callers that
    want a NumPy array can wrap the result. This binary view is the natural
    input to an embedding-based parametrisation of the subsplit probabilities.
    """
    return [(clade >> i) & 1 for i in range(taxon_count)]


def binary_vector_to_clade(vector: tp.Sequence[int]) -> int:
    """Inverse of :func:`clade_to_binary_vector`."""
    clade = 0
    for i, bit in enumerate(vector):
        if bit:
            clade |= 1 << i
    return clade


def make_subsplit(child_a: int, child_b: int) -> Subsplit:
    """Build a canonical :class:`Subsplit` from two disjoint child clades.

    The child containing the smallest-indexed taxon of the union becomes
    ``child1``. Raises ``ValueError`` if the children overlap or either is empty.
    """
    if child_a == 0 or child_b == 0:
        raise ValueError("Subsplit children must be non-empty")
    if child_a & child_b:
        raise ValueError("Subsplit children must be disjoint")
    if min_taxon(child_a) < min_taxon(child_b):
        return Subsplit(child_a, child_b)
    return Subsplit(child_b, child_a)


def enumerate_clade_subsplits(clade: int) -> tp.List[Subsplit]:
    """All canonical subsplits of ``clade``.

    A clade of size ``k`` has ``2 ** (k - 1) - 1`` subsplits. The smallest-indexed
    taxon of the clade is always placed in ``child1``, which both enforces the
    canonical ordering and avoids double-counting the unordered pair. A singleton
    (leaf) clade has no subsplits.
    """
    if is_singleton(clade) or clade == 0:
        return []
    taxa = clade_taxa(clade)
    smallest = taxa[0]
    rest = taxa[1:]
    k = len(rest)
    smallest_bit = 1 << smallest
    subsplits = []
    # Each subset of ``rest`` joins ``smallest`` in child1; child2 gets the
    # complement. Excluding the full subset keeps child2 non-empty.
    for mask in range((1 << k) - 1):
        child1 = smallest_bit
        for j, taxon in enumerate(rest):
            if mask & (1 << j):
                child1 |= 1 << taxon
        child2 = clade ^ child1
        subsplits.append(Subsplit(child1, child2))
    return subsplits


__all__ = [
    "Subsplit",
    "popcount",
    "clade_size",
    "is_singleton",
    "min_taxon",
    "clade_taxa",
    "taxa_to_clade",
    "full_clade",
    "clade_to_binary_vector",
    "binary_vector_to_clade",
    "make_subsplit",
    "enumerate_clade_subsplits",
]
