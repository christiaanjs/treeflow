"""Enumeration and indexing of the conditional-clade / subsplit support.

:class:`ConditionalCladeSupport` enumerates, for a fixed (small) taxon set, every
clade that can be split (size >= 2) and, for each, every subsplit. This is the
*support* of a conditional clade distribution (a subsplit Bayesian network): the
set of conditional events ``P(subsplit | parent clade)`` that the distribution
places probability mass on.

For ``n`` taxa there are ``2**n - n - 1`` splittable clades and the total number
of subsplits across them grows quickly, so this exhaustive representation is
intended for small ``n`` (say up to ~10) where exact distributions over
topologies are tractable. The number of distinct rooted topologies is the double
factorial ``(2n - 3)!!``.

The class also provides:

* a flat indexing of all ``(parent clade, subsplit)`` pairs, with a
  ``segment id`` per pair giving its parent clade -- this lays out a single
  parameter (logit) vector over which a per-parent-clade softmax yields the
  conditional probabilities;
* conversion between a *subsplit assignment* (a choice of one subsplit per
  splittable clade reachable from the root) and TreeFlow's ``parent_indices``
  topology encoding, in both directions;
* exhaustive enumeration of topologies; and
* binary-vector feature matrices for clades and subsplits, the hook for an
  embedding-based parametrisation.
"""

from __future__ import annotations

import typing as tp

import numpy as np

from treeflow.conditional_clade.clade import (
    Subsplit,
    clade_size,
    clade_taxa,
    clade_to_binary_vector,
    enumerate_clade_subsplits,
    full_clade,
    is_singleton,
    make_subsplit,
    min_taxon,
)

# A subsplit assignment maps each splittable clade (bitset) that appears in a
# tree to the subsplit chosen at its node.
SubsplitAssignment = tp.Dict[int, Subsplit]


class ConditionalCladeSupport:
    """The enumerable support of a conditional clade distribution on ``n`` taxa."""

    def __init__(
        self,
        taxon_count: int,
        taxon_set: tp.Optional[tp.Sequence[str]] = None,
    ):
        if taxon_count < 2:
            raise ValueError("Need at least two taxa")
        self.taxon_count = taxon_count
        self.taxon_set = tuple(taxon_set) if taxon_set is not None else None
        if self.taxon_set is not None and len(self.taxon_set) != taxon_count:
            raise ValueError("taxon_set length does not match taxon_count")

        self.root_clade = full_clade(taxon_count)

        # Splittable clades (size >= 2), ordered by decreasing size then bitset.
        # Decreasing size makes the root first and guarantees that any clade is
        # listed before the (strictly smaller) children it can produce -- the
        # ordering the visitation dynamic program relies on.
        parent_clades = [
            clade
            for clade in range(1, 1 << taxon_count)
            if clade_size(clade) >= 2
        ]
        parent_clades.sort(key=lambda c: (-clade_size(c), c))
        self.parent_clades: tp.Tuple[int, ...] = tuple(parent_clades)
        self.parent_clade_index: tp.Dict[int, int] = {
            clade: i for i, clade in enumerate(self.parent_clades)
        }

        # Subsplits per parent clade and the flat layout over all of them.
        self.subsplits_by_parent: tp.List[tp.List[Subsplit]] = []
        self._flat_subsplits: tp.List[Subsplit] = []
        self._flat_parents: tp.List[int] = []
        self._segment_ids: tp.List[int] = []
        # (parent clade, subsplit) -> flat index
        self.flat_index: tp.Dict[tp.Tuple[int, Subsplit], int] = {}
        # per parent clade: start offset into the flat arrays
        self.parent_offsets: tp.List[int] = []
        for parent_idx, clade in enumerate(self.parent_clades):
            subsplits = enumerate_clade_subsplits(clade)
            self.parent_offsets.append(len(self._flat_subsplits))
            self.subsplits_by_parent.append(subsplits)
            for subsplit in subsplits:
                self.flat_index[(clade, subsplit)] = len(self._flat_subsplits)
                self._flat_subsplits.append(subsplit)
                self._flat_parents.append(clade)
                self._segment_ids.append(parent_idx)
        self.subsplit_count = len(self._flat_subsplits)
        self.segment_ids = np.asarray(self._segment_ids, dtype=np.int32)

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------
    @property
    def parent_clade_count(self) -> int:
        return len(self.parent_clades)

    @property
    def flat_subsplits(self) -> tp.Tuple[Subsplit, ...]:
        """All subsplits in flat order (parallel to a logit vector)."""
        return tuple(self._flat_subsplits)

    @property
    def flat_parents(self) -> tp.Tuple[int, ...]:
        """Parent clade (bitset) of each flat subsplit."""
        return tuple(self._flat_parents)

    def topology_count(self) -> int:
        """Number of distinct rooted topologies: ``(2n - 3)!!``."""
        n = self.taxon_count
        result = 1
        for k in range(3, 2 * n - 1, 2):
            result *= k
        return result

    # ------------------------------------------------------------------
    # Conversion: subsplit assignment <-> parent_indices topology
    # ------------------------------------------------------------------
    def assignment_to_parent_indices(
        self, assignment: SubsplitAssignment
    ) -> np.ndarray:
        """Convert a subsplit assignment into TreeFlow ``parent_indices``.

        Follows TreeFlow's labelling convention: leaves take indices
        ``0..n-1`` (the taxon index), internal nodes ``n..2n-2`` with the root
        at ``2n-2``. Internal nodes are numbered in post-order so that every
        node's parent has a strictly larger index, matching the convention used
        by the rooted-tree machinery.

        ``parent_indices`` has length ``2n - 2`` (every node but the root).
        """
        n = self.taxon_count
        parent_indices = np.full(2 * n - 2, -1, dtype=np.int32)
        next_internal = [n]
        node_id: tp.Dict[int, int] = {}

        def assign(clade: int) -> int:
            if is_singleton(clade):
                nid = min_taxon(clade)  # leaf -> taxon index
                node_id[clade] = nid
                return nid
            subsplit = assignment[clade]
            id1 = assign(subsplit.child1)
            id2 = assign(subsplit.child2)
            nid = next_internal[0]
            next_internal[0] += 1
            node_id[clade] = nid
            parent_indices[id1] = nid
            parent_indices[id2] = nid
            return nid

        root_id = assign(self.root_clade)
        assert root_id == 2 * n - 2, "root should be the last node index"
        return parent_indices

    def parent_indices_to_assignment(
        self, parent_indices: tp.Sequence[int]
    ) -> SubsplitAssignment:
        """Recover the subsplit assignment of a ``parent_indices`` topology.

        Computes the clade at each node by a post-order union of its children's
        clades (leaf ``i`` has clade ``{i}``), then reads off the subsplit at
        every internal node. Inverse of :meth:`assignment_to_parent_indices`.
        """
        parent_indices = np.asarray(parent_indices, dtype=np.int64)
        n = self.taxon_count
        node_count = 2 * n - 1
        if parent_indices.shape[-1] != node_count - 1:
            raise ValueError(
                f"Expected parent_indices of length {node_count - 1}, "
                f"got {parent_indices.shape[-1]}"
            )

        children: tp.List[tp.List[int]] = [[] for _ in range(node_count)]
        for child, parent in enumerate(parent_indices):
            children[int(parent)].append(child)

        clade_of = [0] * node_count
        for i in range(n):
            clade_of[i] = 1 << i

        assignment: SubsplitAssignment = {}
        # Internal nodes n..2n-2; post-order numbering guarantees children are
        # processed first, so ascending index order is a valid post-order.
        for node in range(n, node_count):
            child_nodes = children[node]
            if len(child_nodes) != 2:
                raise ValueError(
                    f"Node {node} has {len(child_nodes)} children; expected a "
                    "bifurcating topology"
                )
            c1, c2 = (clade_of[c] for c in child_nodes)
            clade_of[node] = c1 | c2
            assignment[clade_of[node]] = make_subsplit(c1, c2)
        return assignment

    # ------------------------------------------------------------------
    # Enumeration of topologies
    # ------------------------------------------------------------------
    def enumerate_assignments(self) -> tp.Iterator[SubsplitAssignment]:
        """Yield every subsplit assignment (i.e. every rooted topology).

        Generated by recursively choosing a subsplit for the root clade and each
        resulting splittable child clade.
        """

        def expand(clade: int) -> tp.Iterator[SubsplitAssignment]:
            if is_singleton(clade):
                yield {}
                return
            for subsplit in enumerate_clade_subsplits(clade):
                for left in expand(subsplit.child1):
                    for right in expand(subsplit.child2):
                        combined = {clade: subsplit}
                        combined.update(left)
                        combined.update(right)
                        yield combined

        yield from expand(self.root_clade)

    def enumerate_parent_indices(self) -> tp.List[np.ndarray]:
        """All rooted topologies as ``parent_indices`` arrays."""
        return [
            self.assignment_to_parent_indices(assignment)
            for assignment in self.enumerate_assignments()
        ]

    # ------------------------------------------------------------------
    # Flat-index helpers (linking an assignment to a parameter vector)
    # ------------------------------------------------------------------
    def assignment_flat_indices(
        self, assignment: SubsplitAssignment
    ) -> np.ndarray:
        """Flat subsplit indices chosen by an assignment.

        The log-probability of the corresponding topology is the sum of the
        per-subsplit conditional log-probabilities at these indices.
        """
        return np.asarray(
            [self.flat_index[(clade, subsplit)] for clade, subsplit in assignment.items()],
            dtype=np.int32,
        )

    # ------------------------------------------------------------------
    # Binary-vector features (embedding-based parametrisation hook)
    # ------------------------------------------------------------------
    def clade_feature_matrix(self) -> np.ndarray:
        """``(parent_clade_count, n)`` binary membership matrix of parent clades."""
        return np.asarray(
            [clade_to_binary_vector(c, self.taxon_count) for c in self.parent_clades],
            dtype=np.float64,
        )

    def subsplit_feature_matrix(self) -> np.ndarray:
        """``(subsplit_count, 3n)`` binary features for every flat subsplit.

        Each row concatenates the membership vectors of the parent clade and the
        two child clades. These features are the intended input to an
        embedding/neural-network parametrisation that produces a logit per
        subsplit without storing one explicitly.
        """
        n = self.taxon_count
        rows = []
        for subsplit, parent in zip(self._flat_subsplits, self._flat_parents):
            rows.append(
                clade_to_binary_vector(parent, n)
                + clade_to_binary_vector(subsplit.child1, n)
                + clade_to_binary_vector(subsplit.child2, n)
            )
        return np.asarray(rows, dtype=np.float64)


__all__ = ["ConditionalCladeSupport", "SubsplitAssignment"]
