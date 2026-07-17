"""Subsplit Bayesian network: differentiable log-probability and sampling.

The subsplit Bayesian network (SBN) is the variational family VBPI places over
rooted tree *topologies*. Over the fixed support (:class:`SubsplitSupport`) the
probability of a topology factorises as a product of conditional clade
probabilities::

    q(T) = prod over internal clades C in T of  q( subsplit_T(C) | C )

Each conditional ``q(. | C)`` is a categorical over ``C``'s candidate child
subsplits, parameterised by free logits (one per candidate, i.e. per entry of
the support's flat candidate arrays). Normalisation is a *segmented* softmax:
the softmax runs within each parent clade's block of the pointer-array
structure.

This module provides:

* :meth:`SubsplitBayesianNetwork.log_prob` -- the differentiable ``log q(T)``
  used by the VIMCO gradient estimator. Given the per-topology candidate indices
  (from :meth:`SubsplitSupport.topology_candidate_indices` or from the sampler)
  it is a gather-and-sum over the segmented log-softmax, so its gradient with
  respect to the logits is exact TensorFlow autodiff.
* :meth:`SubsplitBayesianNetwork.sample_topologies` -- ancestral sampling of
  whole topologies by walking the pointer-array structure from the root clade
  down, drawing a child subsplit at each clade. A pure-NumPy reference sampler
  is always available; ``use_native`` routes the sequential walk through the
  compiled C++ op in :mod:`treeflow.acceleration.native.sbn`.

Sampling itself is not reparameterised (topologies are discrete); gradients for
the topology parameters flow only through :meth:`log_prob`, which is why VIMCO
is used downstream.
"""
import typing as tp

import numpy as np
import tensorflow as tf

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.vbpi.support import SubsplitSupport


class SampledTopologies(tp.NamedTuple):
    """Output of :meth:`SubsplitBayesianNetwork.sample_topologies`.

    Attributes
    ----------
    parent_indices
        ``int32`` array ``[sample, 2n-2]`` -- treeflow topology per sample.
    candidate_indices
        ``int32`` array ``[sample, n-1]`` -- the candidate/parameter index
        chosen at each internal node (aligned with internal node ids, i.e. entry
        ``k`` is node ``n + k``). Feeding this straight to :meth:`log_prob`
        yields ``log q(T)`` of the sampled trees.
    node_clade_ids
        ``int32`` array ``[sample, 2n-1]`` -- the support clade id of every
        node, used to gather per-split branch-length parameters.
    """

    parent_indices: np.ndarray
    candidate_indices: np.ndarray
    node_clade_ids: np.ndarray


class SubsplitBayesianNetwork:
    """SBN over rooted topologies with a fixed :class:`SubsplitSupport`."""

    def __init__(
        self,
        support: SubsplitSupport,
        logits: tp.Optional[tf.Variable] = None,
        dtype: tf.DType = DEFAULT_FLOAT_DTYPE_TF,
    ):
        self.support = support
        self.dtype = dtype
        if logits is None:
            logits = tf.Variable(
                tf.zeros([support.num_candidates], dtype=dtype),
                name="sbn_logits",
            )
        self.logits = logits
        # Segment structure for the within-parent softmax. Segment ids are the
        # parent clade ids of each candidate, non-decreasing because the support
        # groups candidates by parent in clade-id order (required by
        # tf.math.segment_*).
        self._segment_ids = tf.constant(
            support.candidate_parent_clade.astype(np.int32)
        )
        self._num_segments = support.num_clades

    # ------------------------------------------------------------------
    # Construction helper
    # ------------------------------------------------------------------
    @classmethod
    def from_topologies(
        cls,
        parent_indices_collection: tp.Sequence[np.ndarray],
        taxon_count: int,
        dtype: tf.DType = DEFAULT_FLOAT_DTYPE_TF,
        pseudo_count: float = 1.0,
    ) -> "SubsplitBayesianNetwork":
        """Build support *and* initialise logits from observed topologies.

        The logits are set to the log of the smoothed empirical candidate counts
        (a simple-average / maximum-likelihood CCD estimate with ``pseudo_count``
        Laplace smoothing), which is the usual VBPI warm start.
        """
        support = SubsplitSupport.from_topologies(
            parent_indices_collection, taxon_count
        )
        counts = np.full(support.num_candidates, float(pseudo_count))
        for parent_indices in parent_indices_collection:
            for j in support.topology_candidate_indices(parent_indices):
                counts[j] += 1.0
        logits = tf.Variable(
            tf.constant(np.log(counts), dtype=dtype), name="sbn_logits"
        )
        return cls(support, logits=logits, dtype=dtype)

    # ------------------------------------------------------------------
    # Conditional probabilities
    # ------------------------------------------------------------------
    def conditional_log_probs(self) -> tf.Tensor:
        """Per-candidate ``log q(subsplit | parent clade)`` (segmented softmax).

        Shape ``[num_candidates]``. Within each parent clade's block these are a
        valid log-categorical (they exponentiate-and-sum to one).
        """
        logits = tf.convert_to_tensor(self.logits, dtype=self.dtype)
        # Segmented log-sum-exp normaliser, gathered back per candidate.
        seg_max = tf.math.segment_max(logits, self._segment_ids)
        seg_max_g = tf.gather(seg_max, self._segment_ids)
        seg_sum = tf.math.segment_sum(
            tf.exp(logits - seg_max_g), self._segment_ids
        )
        log_norm = tf.math.log(seg_sum) + seg_max  # per segment
        log_norm_g = tf.gather(log_norm, self._segment_ids)
        return logits - log_norm_g

    def conditional_probs(self) -> tf.Tensor:
        return tf.exp(self.conditional_log_probs())

    # ------------------------------------------------------------------
    # Log probability
    # ------------------------------------------------------------------
    def log_prob(self, candidate_indices: tf.Tensor) -> tf.Tensor:
        """``log q(T)`` from per-tree candidate indices.

        Parameters
        ----------
        candidate_indices
            Integer tensor ``[..., n-1]`` of candidate/parameter indices, one per
            internal node -- as returned by
            :meth:`SubsplitSupport.topology_candidate_indices` or by the sampler.

        Returns
        -------
        Tensor ``[...]`` of topology log-probabilities, differentiable in the
        logits.
        """
        candidate_indices = tf.convert_to_tensor(candidate_indices)
        cond_log_probs = self.conditional_log_probs()
        gathered = tf.gather(cond_log_probs, candidate_indices)
        return tf.reduce_sum(gathered, axis=-1)

    def log_prob_of_topologies(
        self, parent_indices_collection: tp.Sequence[np.ndarray]
    ) -> tf.Tensor:
        """``log q(T)`` for topologies given as ``parent_indices`` arrays."""
        candidate_indices = self.support.batch_topology_candidate_indices(
            parent_indices_collection
        )
        return self.log_prob(tf.constant(candidate_indices))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_topologies(
        self,
        n_samples: int,
        seed: tp.Optional[int] = None,
        use_native: tp.Union[bool, str] = False,
    ) -> SampledTopologies:
        """Ancestral-sample whole topologies from the SBN.

        Parameters
        ----------
        n_samples
            Number of topologies to draw.
        seed
            Base RNG seed (reproducible per backend; the NumPy and native
            backends use independent RNGs so identical seeds do not give
            identical draws across backends).
        use_native
            ``False`` (default) uses the NumPy reference walk; ``True`` requires
            the compiled C++ sampler; ``"auto"`` uses it when available and
            falls back to NumPy otherwise.
        """
        probs = self.conditional_probs().numpy().astype(np.float64)
        native = self._resolve_native(use_native)
        if native:
            return self._sample_native(n_samples, probs, seed)
        return self._sample_numpy(n_samples, probs, seed)

    def _resolve_native(self, use_native: tp.Union[bool, str]) -> bool:
        if use_native == "auto":
            from treeflow.acceleration.native import sbn as native_sbn

            return native_sbn.is_available()
        if isinstance(use_native, bool):
            return use_native
        raise ValueError(
            f"use_native must be True, False or 'auto', got {use_native!r}"
        )

    def _sample_native(
        self, n_samples: int, probs: np.ndarray, seed: tp.Optional[int]
    ) -> SampledTopologies:
        from treeflow.acceleration.native import sbn as native_sbn

        s = self.support
        parent_indices, candidate_indices, node_clade_ids = native_sbn.sample(
            child_offsets=s.child_offsets,
            candidate_left_clade=s.candidate_left_clade,
            candidate_right_clade=s.candidate_right_clade,
            candidate_probs=probs,
            clade_leaf_taxon=s.clade_leaf_taxon,
            root_clade_id=s.root_clade_id,
            taxon_count=s.taxon_count,
            n_samples=n_samples,
            seed=0 if seed is None else int(seed),
        )
        return SampledTopologies(
            parent_indices=parent_indices,
            candidate_indices=candidate_indices,
            node_clade_ids=node_clade_ids,
        )

    def _sample_numpy(
        self, n_samples: int, probs: np.ndarray, seed: tp.Optional[int]
    ) -> SampledTopologies:
        s = self.support
        rng = np.random.default_rng(seed)
        n = s.taxon_count
        node_count = 2 * n - 1
        offsets = s.child_offsets
        left = s.candidate_left_clade
        right = s.candidate_right_clade
        leaf_taxon = s.clade_leaf_taxon

        parent_out = np.empty((n_samples, node_count - 1), dtype=np.int32)
        candidate_out = np.empty((n_samples, n - 1), dtype=np.int32)
        node_clade_out = np.empty((n_samples, node_count), dtype=np.int32)

        for sample in range(n_samples):
            parent = np.full(node_count - 1, -1, dtype=np.int32)
            candidate = np.empty(n - 1, dtype=np.int32)
            node_clade = np.empty(node_count, dtype=np.int32)
            next_internal = [n]

            def visit(clade_id: int) -> int:
                taxon = leaf_taxon[clade_id]
                if taxon >= 0:  # leaf clade -> leaf node
                    node_clade[taxon] = clade_id
                    return int(taxon)
                start = int(offsets[clade_id])
                end = int(offsets[clade_id + 1])
                group = probs[start:end]
                total = group.sum()
                # Renormalise defensively (probs already normalised in exact
                # arithmetic, but guard against tiny drift).
                choice = rng.choice(end - start, p=group / total)
                cand = start + choice
                left_node = visit(int(left[cand]))
                right_node = visit(int(right[cand]))
                my_id = next_internal[0]
                next_internal[0] += 1
                parent[left_node] = my_id
                parent[right_node] = my_id
                candidate[my_id - n] = cand
                node_clade[my_id] = clade_id
                return my_id

            root_id = visit(s.root_clade_id)
            assert root_id == node_count - 1
            parent_out[sample] = parent
            candidate_out[sample] = candidate
            node_clade_out[sample] = node_clade

        return SampledTopologies(
            parent_indices=parent_out,
            candidate_indices=candidate_out,
            node_clade_ids=node_clade_out,
        )

    def __repr__(self) -> str:
        return f"SubsplitBayesianNetwork(support={self.support!r})"


__all__ = ["SubsplitBayesianNetwork", "SampledTopologies"]
