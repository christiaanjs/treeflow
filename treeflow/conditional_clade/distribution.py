"""A conditional clade (subsplit Bayesian network) distribution over topologies.

The distribution factorises the probability of a rooted topology into a product
of per-internal-node conditional probabilities ``P(subsplit | parent clade)``::

    P(tree) = prod over internal nodes of  P(subsplit_v | clade_v)

It is parametrised by a single flat vector of *logits*, one per
``(parent clade, subsplit)`` pair in a :class:`ConditionalCladeSupport`. The
conditional probabilities are obtained by a softmax over the subsplits of each
parent clade (a per-parent *segment* of the flat vector).

The logits can be supplied directly (the exact, table-based parametrisation used
for small taxon sets), or produced by any TensorFlow computation -- for example
an embedding network applied to
:meth:`ConditionalCladeSupport.subsplit_feature_matrix` -- since the class only
ever consumes the logit tensor.

Everything here is differentiable in the logits, which is what makes the gradient
estimator comparison in the accompanying notebook possible. Sampling itself is
discrete; the :mod:`treeflow.conditional_clade.estimators` module supplies the
gradient estimators (score function, leave-one-out / VIMCO, straight-through
Gumbel-Softmax) and the "1/0 probability gradient" sampler.
"""

from __future__ import annotations

import typing as tp

import numpy as np
import tensorflow as tf

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.conditional_clade.clade import is_singleton
from treeflow.conditional_clade.support import (
    ConditionalCladeSupport,
    SubsplitAssignment,
)
from treeflow.conditional_clade.traversal_estimators import (
    traversal_log_prob,
    straight_through_traversal_log_prob,
)
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology


def segment_log_softmax(
    logits: tf.Tensor, segment_ids: tf.Tensor, num_segments: int
) -> tf.Tensor:
    """Log-softmax applied independently within each segment.

    ``segment_ids[i]`` gives the segment of ``logits[i]``; the returned tensor
    has, in each segment, the log of a probability distribution that sums to one.
    Differentiable in ``logits``.
    """
    seg_max = tf.math.unsorted_segment_max(logits, segment_ids, num_segments)
    shifted = logits - tf.gather(seg_max, segment_ids)
    seg_sumexp = tf.math.unsorted_segment_sum(
        tf.exp(shifted), segment_ids, num_segments
    )
    return shifted - tf.gather(tf.math.log(seg_sumexp), segment_ids)


class ConditionalCladeDistribution:
    """Distribution over rooted topologies defined by conditional clade logits."""

    def __init__(
        self,
        support: ConditionalCladeSupport,
        logits: tp.Optional[tf.Tensor] = None,
        dtype: tf.DType = DEFAULT_FLOAT_DTYPE_TF,
    ):
        self.support = support
        self.dtype = dtype
        if logits is None:
            logits = tf.zeros([support.subsplit_count], dtype=dtype)
        # Keep the original object (which may be a tf.Variable) rather than
        # eagerly converting: converting here would snapshot a Variable outside
        # any GradientTape and break differentiability. Conversion happens
        # lazily inside ``conditional_log_probs`` instead.
        self.logits = logits
        if int(logits.shape[-1]) != support.subsplit_count:
            raise ValueError(
                f"Expected {support.subsplit_count} logits, "
                f"got {int(logits.shape[-1])}"
            )
        self._segment_ids = tf.constant(support.segment_ids, dtype=tf.int32)
        self._num_segments = support.parent_clade_count
        self._enumeration_cache: tp.Optional[
            tp.Tuple[tp.List[np.ndarray], tf.Tensor]
        ] = None

    # ------------------------------------------------------------------
    # Conditional probabilities
    # ------------------------------------------------------------------
    def conditional_log_probs(self) -> tf.Tensor:
        """Per-subsplit conditional log-probabilities (flat, length ``M``)."""
        logits = tf.convert_to_tensor(self.logits, dtype=self.dtype)
        return segment_log_softmax(logits, self._segment_ids, self._num_segments)

    def conditional_probs_numpy(self) -> np.ndarray:
        """Conditional probabilities as a NumPy array (for sampling/analysis)."""
        return np.exp(self.conditional_log_probs().numpy())

    # ------------------------------------------------------------------
    # Log-probability
    # ------------------------------------------------------------------
    def log_prob_from_flat_indices(self, flat_indices: tf.Tensor) -> tf.Tensor:
        """Log-probability of a topology given its chosen flat subsplit indices.

        ``flat_indices`` may carry a leading batch dimension (one row per tree);
        the conditional log-probabilities are summed over the last axis. This is
        the exact log-prob of a pre-sampled traversal -- the score-function
        gradient used by REINFORCE / leave-one-out / VIMCO.
        """
        return traversal_log_prob(self.conditional_log_probs(), flat_indices)

    def straight_through_log_prob_from_flat_indices(
        self,
        flat_indices: tf.Tensor,
        temperature: float = 1.0,
        gumbel: bool = False,
        seed=None,
    ) -> tf.Tensor:
        """Straight-through ``log q(T)`` for a pre-sampled traversal.

        Forward value equals :meth:`log_prob_from_flat_indices`; the gradient
        flows through the per-clade (Gumbel-)softmax relaxation. Vectorised graph
        ops, so it runs inside ``tf.function`` with no per-node Python recursion.
        """
        return straight_through_traversal_log_prob(
            self.conditional_log_probs(),
            tf.convert_to_tensor(self.logits, self.dtype),
            flat_indices,
            self._segment_ids,
            temperature=temperature,
            gumbel=gumbel,
            seed=seed,
        )

    def log_prob_assignment(self, assignment: SubsplitAssignment) -> tf.Tensor:
        flat_indices = self.support.assignment_flat_indices(assignment)
        return self.log_prob_from_flat_indices(tf.constant(flat_indices, tf.int32))

    def log_prob(self, parent_indices: tp.Sequence[int]) -> tf.Tensor:
        """Log-probability of a topology given as ``parent_indices``."""
        assignment = self.support.parent_indices_to_assignment(parent_indices)
        return self.log_prob_assignment(assignment)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _resolve_rng(self, seed) -> np.random.Generator:
        if isinstance(seed, np.random.Generator):
            return seed
        return np.random.default_rng(seed)

    def sample_assignment(self, seed=None) -> SubsplitAssignment:
        """Sample a topology, returned as a subsplit assignment."""
        rng = self._resolve_rng(seed)
        cond = self.conditional_probs_numpy()
        support = self.support
        assignment: SubsplitAssignment = {}

        def expand(clade: int) -> None:
            if is_singleton(clade):
                return
            parent_idx = support.parent_clade_index[clade]
            start = support.parent_offsets[parent_idx]
            subsplits = support.subsplits_by_parent[parent_idx]
            probs = cond[start : start + len(subsplits)]
            # Guard against tiny normalisation error from the softmax.
            probs = probs / probs.sum()
            choice = rng.choice(len(subsplits), p=probs)
            subsplit = subsplits[choice]
            assignment[clade] = subsplit
            expand(subsplit.child1)
            expand(subsplit.child2)

        expand(support.root_clade)
        return assignment

    def sample_parent_indices(self, seed=None) -> np.ndarray:
        """Sample a topology as a ``parent_indices`` array."""
        return self.support.assignment_to_parent_indices(
            self.sample_assignment(seed)
        )

    def sample_topology(self, seed=None) -> NumpyTreeTopology:
        """Sample a topology as a :class:`NumpyTreeTopology`."""
        return NumpyTreeTopology(
            parent_indices=self.sample_parent_indices(seed),
            taxon_set=self.support.taxon_set,
        )

    def sample_flat_index_batch(self, n: int, seed=None) -> np.ndarray:
        """Sample ``n`` topologies as an ``(n, taxon_count - 1)`` index matrix.

        Each row holds the flat subsplit indices of one sampled topology, ready
        to feed :meth:`log_prob_from_flat_indices`. This is the building block of
        the Monte-Carlo gradient estimators.
        """
        rng = self._resolve_rng(seed)
        rows = []
        for _ in range(n):
            assignment = self.sample_assignment(rng)
            rows.append(self.support.assignment_flat_indices(assignment))
        return np.asarray(rows, dtype=np.int32)

    # ------------------------------------------------------------------
    # Exhaustive enumeration (small taxon sets)
    # ------------------------------------------------------------------
    def _enumeration(self) -> tp.Tuple[tp.List[np.ndarray], np.ndarray]:
        """Cache of (parent_indices list, ``(num_trees, n-1)`` flat-index matrix)."""
        if self._enumeration_cache is None:
            parent_indices_list = []
            index_rows = []
            for assignment in self.support.enumerate_assignments():
                parent_indices_list.append(
                    self.support.assignment_to_parent_indices(assignment)
                )
                index_rows.append(self.support.assignment_flat_indices(assignment))
            index_matrix = np.asarray(index_rows, dtype=np.int32)
            self._enumeration_cache = (parent_indices_list, index_matrix)
        return self._enumeration_cache

    def enumerate_parent_indices(self) -> tp.List[np.ndarray]:
        return list(self._enumeration()[0])

    def enumerate_log_probs(self) -> tf.Tensor:
        """Log-probabilities of every topology, in enumeration order.

        Differentiable in the logits. ``exp`` of the result sums to one.
        """
        _, index_matrix = self._enumeration()
        return self.log_prob_from_flat_indices(tf.constant(index_matrix, tf.int32))

    def enumerate_probs(self) -> tf.Tensor:
        return tf.exp(self.enumerate_log_probs())

    def entropy(self) -> tf.Tensor:
        """Exact entropy ``-sum_T q(T) log q(T)`` (enumerated)."""
        log_probs = self.enumerate_log_probs()
        return -tf.reduce_sum(tf.exp(log_probs) * log_probs)

    def exact_kl_divergence(
        self, other: "ConditionalCladeDistribution"
    ) -> tf.Tensor:
        """Exact ``KL(self || other)`` by summing over all topologies.

        Both distributions must share the same support. Differentiable in the
        logits of both, providing an exact-gradient reference for the
        stochastic estimators.
        """
        if other.support is not self.support:
            raise ValueError("KL requires the two distributions to share a support")
        log_q = self.enumerate_log_probs()
        log_p = other.enumerate_log_probs()
        return tf.reduce_sum(tf.exp(log_q) * (log_q - log_p))

    # ------------------------------------------------------------------
    # Clade visitation (dynamic program, for analysis)
    # ------------------------------------------------------------------
    def clade_visitation_probabilities(self) -> tp.Dict[int, float]:
        """Probability that each splittable clade appears as a node in a sample.

        Computed by a top-down dynamic program over the parent clades (which are
        ordered largest-first, so a clade is always processed before any clade it
        can produce). Useful for diagnostics and for a factored KL, and scales far
        better than enumerating topologies.
        """
        cond = self.conditional_probs_numpy()
        support = self.support
        visit = {clade: 0.0 for clade in support.parent_clades}
        visit[support.root_clade] = 1.0
        for parent_idx, clade in enumerate(support.parent_clades):
            v = visit[clade]
            if v == 0.0:
                continue
            start = support.parent_offsets[parent_idx]
            subsplits = support.subsplits_by_parent[parent_idx]
            probs = cond[start : start + len(subsplits)]
            for subsplit, prob in zip(subsplits, probs):
                for child in (subsplit.child1, subsplit.child2):
                    if not is_singleton(child):
                        visit[child] += v * float(prob)
        return visit


__all__ = ["ConditionalCladeDistribution", "segment_log_softmax"]
