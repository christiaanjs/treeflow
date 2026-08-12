"""Pointer-array (CSR) data structure for a subsplit Bayesian network's support.

VBPI parameterises the topology posterior by a *conditional clade distribution*
(CCD / subsplit Bayesian network). The set of clades and, for each clade, the
set of candidate child subsplits it may resolve into -- the network's
*support* -- is fixed up front (typically from a bootstrap or MCMC collection of
trees). Only the conditional probabilities are learned.

:class:`SubsplitSupport` stores that fixed support as flat integer arrays in a
compressed-sparse-row (CSR) layout, exactly the "pointer array based data
structure" the SBN's probabilities live on:

* every distinct clade is assigned an integer id (leaves included);
* ``child_offsets`` (length ``num_clades + 1``) delimits, for each clade id,
  the contiguous block of its candidate child subsplits inside the flat
  candidate arrays -- clade ``c``'s candidates are
  ``[child_offsets[c], child_offsets[c + 1])``;
* ``candidate_left_clade`` / ``candidate_right_clade`` give the two child clade
  ids of each candidate;
* ``candidate_parent_clade`` is the reverse pointer (candidate -> its parent
  clade id), used to normalise the conditional probabilities with a segmented
  softmax.

A leaf clade has an empty candidate block, so the CSR is uniform over all clade
ids. The number of candidates equals the number of free conditional-probability
parameters of the SBN.

Both the differentiable TensorFlow log-probability
(:mod:`treeflow.vbpi.sbn`) and the native C++ sampler
(:mod:`treeflow.acceleration.native.sbn`) consume these arrays directly; the
clade/subsplit *objects* never cross into TensorFlow or C++.
"""
import typing as tp

import numpy as np

from treeflow.vbpi.clade import (
    Clade,
    Subsplit,
    canonical_subsplit,
    clade_sort_key,
    decompose_topology,
    is_leaf_clade,
    root_clade,
)


def _subsplit_key(subsplit: Subsplit) -> tp.Tuple:
    """Hashable, order-independent key identifying a subsplit."""
    left, right = subsplit
    return (clade_sort_key(left), clade_sort_key(right))


class SubsplitSupport:
    """Fixed support of an SBN as CSR pointer arrays over clades and subsplits.

    Construct with :meth:`from_topologies`. The public array attributes are all
    plain NumPy ``int`` arrays suitable for handing to TensorFlow gathers or the
    native op.
    """

    def __init__(
        self,
        taxon_count: int,
        clades: tp.Sequence[Clade],
        candidates: tp.Sequence[tp.Tuple[int, Subsplit]],
    ):
        """Low-level constructor; prefer :meth:`from_topologies`.

        Parameters
        ----------
        taxon_count
            Number of leaf taxa ``n``.
        clades
            The distinct clades, already ordered; index in this sequence becomes
            the clade id.
        candidates
            One entry per candidate child subsplit, as
            ``(parent_clade_id, subsplit)``, grouped so that all candidates of a
            given parent clade are contiguous and parents appear in clade-id
            order.
        """
        self.taxon_count = int(taxon_count)
        self._clades: tp.List[Clade] = list(clades)
        self._clade_to_id: tp.Dict[Clade, int] = {
            clade: i for i, clade in enumerate(self._clades)
        }
        num_clades = len(self._clades)
        self.num_clades = num_clades

        self._root_clade = root_clade(self.taxon_count)
        self.root_clade_id = self._clade_to_id[self._root_clade]

        # Flat candidate arrays + CSR offsets.
        num_candidates = len(candidates)
        self.num_candidates = num_candidates
        candidate_parent = np.empty(num_candidates, dtype=np.int32)
        candidate_left = np.empty(num_candidates, dtype=np.int32)
        candidate_right = np.empty(num_candidates, dtype=np.int32)
        counts = np.zeros(num_clades, dtype=np.int64)
        # Map (parent_id, subsplit_key) -> candidate index for fast lookup.
        self._candidate_index: tp.Dict[tp.Tuple[int, tp.Tuple], int] = {}
        for j, (parent_id, subsplit) in enumerate(candidates):
            left, right = subsplit
            candidate_parent[j] = parent_id
            candidate_left[j] = self._clade_to_id[left]
            candidate_right[j] = self._clade_to_id[right]
            counts[parent_id] += 1
            self._candidate_index[(parent_id, _subsplit_key(subsplit))] = j
        offsets = np.zeros(num_clades + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        self.child_offsets = offsets.astype(np.int64)
        self.candidate_parent_clade = candidate_parent
        self.candidate_left_clade = candidate_left
        self.candidate_right_clade = candidate_right

        # Per-clade leaf taxon (or -1); used by the sampler to build topologies.
        leaf_taxon = np.full(num_clades, -1, dtype=np.int32)
        for clade, cid in self._clade_to_id.items():
            if is_leaf_clade(clade):
                (taxon,) = tuple(clade)
                leaf_taxon[cid] = taxon
        self.clade_leaf_taxon = leaf_taxon

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_topologies(
        cls,
        parent_indices_collection: tp.Iterable[np.ndarray],
        taxon_count: int,
    ) -> "SubsplitSupport":
        """Build the support spanning a collection of rooted topologies.

        Every ``(parent clade, child subsplit)`` pair observed in any of the
        input topologies becomes a candidate; every clade observed (including
        leaves and the root) becomes a clade id. Clades are ordered by
        :func:`clade_sort_key` and candidates by their subsplit key, so the
        resulting ids are deterministic and independent of input order.
        """
        taxon_count = int(taxon_count)
        # parent clade -> set of subsplits, plus the global clade set.
        clade_set: tp.Set[Clade] = set(
            frozenset((i,)) for i in range(taxon_count)
        )
        clade_set.add(root_clade(taxon_count))
        parent_to_subsplits: tp.Dict[Clade, tp.Dict[tp.Tuple, Subsplit]] = {}
        for parent_indices in parent_indices_collection:
            for parent_clade, subsplit in decompose_topology(
                np.asarray(parent_indices), taxon_count
            ):
                clade_set.add(parent_clade)
                left, right = subsplit
                clade_set.add(left)
                clade_set.add(right)
                bucket = parent_to_subsplits.setdefault(parent_clade, {})
                bucket[_subsplit_key(subsplit)] = subsplit

        clades = sorted(clade_set, key=clade_sort_key)
        clade_to_id = {clade: i for i, clade in enumerate(clades)}

        candidates: tp.List[tp.Tuple[int, Subsplit]] = []
        # Emit candidates grouped by parent clade in clade-id order.
        for clade in clades:
            subsplits = parent_to_subsplits.get(clade)
            if not subsplits:
                continue
            for key in sorted(subsplits.keys()):
                candidates.append((clade_to_id[clade], subsplits[key]))
        return cls(taxon_count, clades, candidates)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def clade_id(self, clade: Clade) -> int:
        return self._clade_to_id[clade]

    def contains_topology(self, parent_indices: np.ndarray) -> bool:
        """Whether every clade/subsplit of a topology is in the support."""
        try:
            self.topology_candidate_indices(parent_indices)
            return True
        except KeyError:
            return False

    def topology_candidate_indices(
        self, parent_indices: np.ndarray
    ) -> np.ndarray:
        """Candidate (parameter) index chosen at each internal clade of a tree.

        Returns an ``int`` array of length ``n - 1`` -- one candidate index per
        internal node -- that indexes the flat candidate/parameter arrays. The
        SBN log-probability of the topology is the sum of the per-candidate
        log conditional probabilities at exactly these indices.

        Raises ``KeyError`` if the topology contains a clade or subsplit outside
        the support.
        """
        indices = np.empty(self.taxon_count - 1, dtype=np.int64)
        for k, (parent_clade, subsplit) in enumerate(
            decompose_topology(np.asarray(parent_indices), self.taxon_count)
        ):
            parent_id = self._clade_to_id[parent_clade]
            indices[k] = self._candidate_index[(parent_id, _subsplit_key(subsplit))]
        return indices

    def batch_topology_candidate_indices(
        self, parent_indices_collection: tp.Iterable[np.ndarray]
    ) -> np.ndarray:
        """Stack :meth:`topology_candidate_indices` over a collection of trees."""
        return np.stack(
            [
                self.topology_candidate_indices(parent_indices)
                for parent_indices in parent_indices_collection
            ],
            axis=0,
        )

    def __repr__(self) -> str:
        return (
            f"SubsplitSupport(taxon_count={self.taxon_count}, "
            f"num_clades={self.num_clades}, "
            f"num_candidates={self.num_candidates})"
        )


def all_rooted_parent_indices(taxon_count: int) -> tp.List[np.ndarray]:
    """Enumerate every rooted binary topology on ``taxon_count`` taxa.

    Returns each as a treeflow ``parent_indices`` array (leaves ``0..n-1``,
    internal nodes assigned in postorder, root last). Grows as the double
    factorial ``(2n-3)!!`` so this is only practical for small ``n`` -- it
    exists mainly to build a *complete* support and to check SBN normalisation
    in tests.
    """
    taxon_count = int(taxon_count)
    if taxon_count < 1:
        raise ValueError("taxon_count must be >= 1")

    # Enumerate topologies as nested frozenset structures, then relabel to
    # treeflow parent_indices. A "shape" is either a taxon int (leaf) or a
    # frozenset of two sub-shapes.
    def shapes(taxa: tp.Tuple[int, ...]):
        if len(taxa) == 1:
            yield taxa[0]
            return
        first = taxa[0]
        rest = taxa[1:]
        # Partition: place ``first`` on one side; iterate over subsets of the
        # rest joining it, from empty (so the other side has all the rest) up.
        m = len(rest)
        # ``first`` always joins the left side; the right side gets a non-empty
        # subset of the rest (mask up to but excluding "all on the left"), which
        # enumerates every unordered bipartition exactly once per shape.
        for mask in range(0, (1 << m) - 1):
            left_rest = tuple(rest[i] for i in range(m) if (mask >> i) & 1)
            right_taxa = tuple(rest[i] for i in range(m) if not ((mask >> i) & 1))
            left_taxa = (first,) + left_rest
            for left_shape in shapes(left_taxa):
                for right_shape in shapes(right_taxa):
                    yield frozenset((left_shape, right_shape))

    taxa = tuple(range(taxon_count))
    seen: tp.Set = set()
    result: tp.List[np.ndarray] = []
    for shape in shapes(taxa):
        if shape in seen:
            continue
        seen.add(shape)
        result.append(_shape_to_parent_indices(shape, taxon_count))
    return result


def _shape_to_parent_indices(shape, taxon_count: int) -> np.ndarray:
    node_count = 2 * taxon_count - 1
    parent = np.full(node_count - 1, -1, dtype=np.int32)
    next_internal = [taxon_count]

    def assign(node) -> int:
        if isinstance(node, int):
            return node
        # frozenset of two sub-shapes; assign children first (postorder).
        children = sorted(node, key=lambda s: _shape_leaf_key(s))
        child_ids = [assign(c) for c in children]
        my_id = next_internal[0]
        next_internal[0] += 1
        for c in child_ids:
            parent[c] = my_id
        return my_id

    root_id = assign(shape)
    assert root_id == node_count - 1
    return parent


def _shape_leaf_key(shape):
    if isinstance(shape, int):
        return (0, shape)
    leaves = _shape_leaves(shape)
    return (1, tuple(sorted(leaves)))


def _shape_leaves(shape) -> tp.List[int]:
    if isinstance(shape, int):
        return [shape]
    out: tp.List[int] = []
    for s in shape:
        out.extend(_shape_leaves(s))
    return out


__all__ = ["SubsplitSupport", "all_rooted_parent_indices"]
