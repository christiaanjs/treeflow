"""Clade and subsplit primitives for subsplit Bayesian networks (SBNs).

This module implements the combinatorial building blocks that VBPI-style
topology inference is defined over. Everything here is pure Python / NumPy and
deals with *rooted* binary tree topologies in treeflow's index convention (see
:mod:`treeflow.tree.topology`): the ``n`` leaves are labelled ``0 .. n-1`` and
the ``n-1`` internal nodes ``n .. 2n-2`` with the root last.

Definitions
-----------
clade
    A non-empty subset of the taxa. Concretely a taxon is an integer in
    ``0 .. n-1`` and a clade is a :class:`frozenset` of taxa. The *root clade*
    is the set of all taxa. A clade of size one is a leaf.

subsplit
    An (unordered) split of a clade ``C`` into the two child clades
    ``{C1, C2}`` with ``C1 | C2 == C`` and ``C1 & C2 == frozenset()``. We store
    it as an *ordered* pair ``(left, right)`` in a canonical order (see
    :func:`canonical_subsplit`) so it hashes/compares deterministically, but two
    subsplits are equal iff they induce the same unordered pair.

The conditional-clade / subsplit factorisation of a rooted topology ``T`` is::

    P(T) = prod over internal clades C in T of  P( subsplit_T(C) | C )

:func:`decompose_topology` extracts exactly the ``(C, subsplit_T(C))`` pairs
that this product ranges over. Downstream, :mod:`treeflow.vbpi.support` collects
these across a set of trees into the flat pointer-array structure the SBN is
parameterised on.
"""
import typing as tp

import numpy as np

# A clade is a frozenset of taxon indices; a subsplit is an ordered pair of
# clades (canonically ordered, see ``canonical_subsplit``).
Clade = tp.FrozenSet[int]
Subsplit = tp.Tuple[Clade, Clade]


def clade_sort_key(clade: Clade) -> tp.Tuple[int, tp.Tuple[int, ...]]:
    """Deterministic ordering key for clades.

    Orders first by size then by the sorted taxa, giving a total order that is
    stable across runs (unlike the hash-based iteration order of a set).
    """
    return (len(clade), tuple(sorted(clade)))


def canonical_subsplit(child_a: Clade, child_b: Clade) -> Subsplit:
    """Return the two child clades as an ordered ``(left, right)`` pair.

    The order is defined by :func:`clade_sort_key` so that the same unordered
    split always produces the same tuple. This canonical orientation is what
    the native sampler emits (``left`` child before ``right`` child), but it
    carries no probabilistic meaning: ``P(subsplit | C)`` is a distribution over
    unordered splits.
    """
    if clade_sort_key(child_a) <= clade_sort_key(child_b):
        return (child_a, child_b)
    return (child_b, child_a)


def child_indices_from_parent_indices(
    parent_indices: np.ndarray, node_count: int
) -> tp.List[tp.List[int]]:
    """Children of every node, as a list of (0, 1 or 2)-element lists.

    ``parent_indices`` follows treeflow's convention: length ``2n-2`` (an entry
    for every node except the root), ``parent_indices[i]`` is the node id of
    ``i``'s parent.
    """
    children: tp.List[tp.List[int]] = [[] for _ in range(node_count)]
    for node, parent in enumerate(parent_indices):
        children[int(parent)].append(node)
    return children


def node_clades(
    parent_indices: np.ndarray, taxon_count: int
) -> tp.List[Clade]:
    """Clade (set of descendant taxa) of every node, indexed by node id.

    Leaves ``0 .. n-1`` map to their singleton ``{i}``; each internal node maps
    to the union of its children's clades. Computed in a single postorder-free
    pass by processing nodes in increasing id order, which is a valid postorder
    because treeflow guarantees children have smaller ids than their parent.
    """
    node_count = 2 * taxon_count - 1
    parent_indices = np.asarray(parent_indices)
    if parent_indices.shape[-1] != node_count - 1:
        raise ValueError(
            "parent_indices must have length 2n-2 = "
            f"{node_count - 1} for {taxon_count} taxa, got "
            f"{parent_indices.shape[-1]}"
        )
    clades: tp.List[tp.Optional[Clade]] = [None] * node_count
    for i in range(taxon_count):
        clades[i] = frozenset((i,))
    children = child_indices_from_parent_indices(parent_indices, node_count)
    # Children always have smaller ids than parents, so ascending id order is a
    # valid postorder: a node's children are finished before it is reached.
    for node in range(taxon_count, node_count):
        child_clades = [clades[c] for c in children[node]]
        if len(child_clades) != 2 or any(c is None for c in child_clades):
            raise ValueError(
                f"Node {node} is not a bifurcation with two resolved children"
            )
        clades[node] = child_clades[0] | child_clades[1]
    return [c for c in clades]  # type: ignore[misc]


def decompose_topology(
    parent_indices: np.ndarray, taxon_count: int
) -> tp.List[tp.Tuple[Clade, Subsplit]]:
    """Decompose a rooted topology into ``(parent clade, child subsplit)`` pairs.

    Returns one entry per internal node: the clade at that node and the
    canonical subsplit of its two child clades. The parent clade of the root is
    the full taxon set, so the first factor in the CCD product is the root
    subsplit distribution. Ordering of the returned list is by ascending node
    id (postorder), which is deterministic but carries no meaning for the CCD
    product (the factors commute).
    """
    node_count = 2 * taxon_count - 1
    clades = node_clades(parent_indices, taxon_count)
    children = child_indices_from_parent_indices(parent_indices, node_count)
    pairs: tp.List[tp.Tuple[Clade, Subsplit]] = []
    for node in range(taxon_count, node_count):
        c0, c1 = children[node]
        subsplit = canonical_subsplit(clades[c0], clades[c1])
        pairs.append((clades[node], subsplit))
    return pairs


def root_clade(taxon_count: int) -> Clade:
    """The clade containing every taxon (the root's clade)."""
    return frozenset(range(taxon_count))


def is_leaf_clade(clade: Clade) -> bool:
    return len(clade) == 1


__all__ = [
    "Clade",
    "Subsplit",
    "clade_sort_key",
    "canonical_subsplit",
    "child_indices_from_parent_indices",
    "node_clades",
    "decompose_topology",
    "root_clade",
    "is_leaf_clade",
]
