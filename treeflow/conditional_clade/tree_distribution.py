"""A tfp ``Distribution`` over rooted topologies (``TensorflowTreeTopology``).

:class:`ConditionalCladeTreeDistribution` wraps the conditional clade model as a
fully fledged ``tensorflow_probability`` distribution whose samples are
:class:`TensorflowTreeTopology` objects (``parent_indices``, ``child_indices``,
``preorder_indices``). Unlike the eager
:class:`~treeflow.conditional_clade.distribution.ConditionalCladeDistribution`,
its ``sample`` and ``log_prob`` are implemented with the graph-compatible tensor
operations in :mod:`treeflow.conditional_clade.tensor_ops`, so they run inside a
``tf.function``.

It composes the eager distribution (exposed as :attr:`ccd`) for the exact
enumeration / KL utilities, and shares its logits, so the two views stay
consistent.
"""

from __future__ import annotations

import typing as tp

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions.distribution import (
    _set_sample_static_shape_for_tensor,
)
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.conditional_clade.distribution import ConditionalCladeDistribution
from treeflow.conditional_clade.support import ConditionalCladeSupport
from treeflow.conditional_clade import tensor_ops
from treeflow.distributions.tree.base_tree_distribution import BaseTreeDistribution
from treeflow.tree.taxon_set import TupleTaxonSet
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology


def _native_conditional_clade():
    """Lazily import the native op module (kept optional)."""
    from treeflow.acceleration.native import conditional_clade

    return conditional_clade


class ConditionalCladeTreeDistribution(BaseTreeDistribution[TensorflowTreeTopology]):
    """Conditional clade distribution over ``TensorflowTreeTopology`` samples."""

    def __init__(
        self,
        support: ConditionalCladeSupport,
        logits: tp.Optional[tf.Tensor] = None,
        float_dtype: tf.DType = DEFAULT_FLOAT_DTYPE_TF,
        use_native: tp.Union[bool, str] = "auto",
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "ConditionalCladeTreeDistribution",
    ):
        parameters = dict(locals())
        self.ccd = ConditionalCladeDistribution(support, logits, dtype=float_dtype)
        self.support = support
        self.float_dtype = float_dtype

        # Route sample / log_prob through the native C++ ops when requested and
        # available; "auto" uses them if the library can be loaded, else falls
        # back to the pure-TensorFlow tensor_ops implementation.
        if use_native == "auto":
            self._use_native = _native_conditional_clade().is_available()
        else:
            self._use_native = bool(use_native)
            if self._use_native:
                _native_conditional_clade().load_op_library()

        n = support.taxon_count
        self._n = n
        self._node_count = 2 * n - 1
        self._pow2n = 1 << n

        # Constant lookup tensors for the graph-mode sampler.
        node_id_init = np.full(self._pow2n, -1, dtype=np.int32)
        for i in range(n):
            node_id_init[1 << i] = i
        clade_offset = np.zeros(self._pow2n, dtype=np.int32)
        clade_count = np.zeros(self._pow2n, dtype=np.int32)
        for parent_idx, clade in enumerate(support.parent_clades):
            clade_offset[clade] = support.parent_offsets[parent_idx]
            clade_count[clade] = len(support.subsplits_by_parent[parent_idx])
        self._node_id_init = tf.constant(node_id_init)
        self._clade_offset = tf.constant(clade_offset)
        self._clade_count = tf.constant(clade_count)
        self._flat_child1 = tf.constant(
            [s.child1 for s in support.flat_subsplits], dtype=tf.int32
        )
        self._flat_child2 = tf.constant(
            [s.child2 for s in support.flat_subsplits], dtype=tf.int32
        )
        self._flat_parent = tf.constant(support.flat_parents, dtype=tf.int32)

        # Hash table (parent * 2**n + canonical child1) -> flat subsplit index,
        # for the graph-mode log-probability.
        keys = [
            parent * self._pow2n + subsplit.child1
            for subsplit, parent in zip(
                support.flat_subsplits, support.flat_parents
            )
        ]
        values = list(range(support.subsplit_count))
        self._flat_index_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(keys, dtype=tf.int64),
                tf.constant(values, dtype=tf.int32),
            ),
            default_value=-1,
        )

        super().__init__(
            taxon_count=n,
            reparameterization_type=BaseTreeDistribution._topology_reparameterization_type,
            dtype=BaseTreeDistribution._topology_dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            parameters=parameters,
        )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    @property
    def logits(self) -> tf.Tensor:
        return self.ccd.logits

    def conditional_log_probs(self) -> tf.Tensor:
        return self.ccd.conditional_log_probs()

    # ------------------------------------------------------------------
    # Event shape
    # ------------------------------------------------------------------
    def _event_shape(self) -> TensorflowTreeTopology:
        return self._topology_event_shape()

    def _event_shape_tensor(self) -> TensorflowTreeTopology:
        return self._topology_event_shape_tensor()

    # ------------------------------------------------------------------
    # Sampling (graph mode)
    # ------------------------------------------------------------------
    def _sample_one(self, seed: tf.Tensor) -> TensorflowTreeTopology:
        logits = tf.convert_to_tensor(self.logits, dtype=self.float_dtype)
        parent_indices = tensor_ops.sample_parent_indices(
            logits,
            node_id_init=self._node_id_init,
            clade_offset=self._clade_offset,
            clade_count=self._clade_count,
            flat_child1=self._flat_child1,
            flat_child2=self._flat_child2,
            taxon_count=self._n,
            node_count=self._node_count,
            seed=seed,
        )
        child_indices = tensor_ops.parent_indices_to_child_indices(
            parent_indices, self._node_count
        )
        preorder_indices = tensor_ops.child_indices_to_preorder(
            child_indices, self._node_count
        )
        return TensorflowTreeTopology(
            parent_indices=parent_indices,
            child_indices=child_indices,
            preorder_indices=preorder_indices,
        )

    def _sample_n_native(self, per_sample_seeds) -> TensorflowTreeTopology:
        native = _native_conditional_clade()
        logits = tf.convert_to_tensor(self.logits, dtype=self.float_dtype)
        parent_indices = native.native_sample_parent_indices(
            logits,
            tf.cast(per_sample_seeds, tf.int32),
            self._clade_offset,
            self._clade_count,
            self._flat_child1,
            self._flat_child2,
            self._n,
        )
        child_indices = native.native_parent_indices_to_child_indices(
            parent_indices, self._n
        )
        preorder_indices = native.native_child_indices_to_preorder(
            child_indices, self._n
        )
        return TensorflowTreeTopology(
            parent_indices=parent_indices,
            child_indices=child_indices,
            preorder_indices=preorder_indices,
        )

    def _sample_n(self, n, seed=None) -> TensorflowTreeTopology:
        seed = samplers.sanitize_seed(seed, salt="ConditionalCladeTreeDistribution")
        per_sample_seeds = samplers.split_seed(
            seed, n=tf.convert_to_tensor(n, dtype=tf.int32)
        )
        if self._use_native:
            return self._sample_n_native(per_sample_seeds)
        output_signature = TensorflowTreeTopology(
            parent_indices=tf.TensorSpec([self._node_count - 1], tf.int32),
            child_indices=tf.TensorSpec([self._node_count, 2], tf.int32),
            preorder_indices=tf.TensorSpec([self._node_count], tf.int32),
        )
        return tf.map_fn(
            self._sample_one,
            per_sample_seeds,
            fn_output_signature=output_signature,
        )

    def _call_sample_n(self, sample_shape, seed, **kwargs) -> TensorflowTreeTopology:
        sample_shape = ps.convert_to_shape_tensor(
            ps.cast(sample_shape, tf.int32), name="sample_shape"
        )
        sample_shape, n = self._expand_sample_shape_to_vector(
            sample_shape, "sample_shape"
        )
        flat_samples = self._sample_n(n, seed=seed() if callable(seed) else seed)
        event_shape = self.event_shape
        batch_shape = self.batch_shape

        def reshape_leaf(leaf, leaf_event_shape):
            final_shape = ps.concat([sample_shape, ps.shape(leaf)[1:]], 0)
            leaf = tf.reshape(leaf, final_shape)
            return _set_sample_static_shape_for_tensor(
                leaf, leaf_event_shape, batch_shape, sample_shape
            )

        return tf.nest.map_structure(reshape_leaf, flat_samples, event_shape)

    # ------------------------------------------------------------------
    # Log-probability (graph mode)
    # ------------------------------------------------------------------
    def _call_log_prob(self, value, name="log_prob", **kwargs) -> tf.Tensor:
        # The log-probability depends only on ``parent_indices``; override the
        # base wrapper so the (optional) ``child_indices`` / ``preorder_indices``
        # fields need not be supplied.
        with self._name_and_control_scope(name):
            return self._log_prob(value)

    def _log_prob(self, value: TensorflowTreeTopology) -> tf.Tensor:
        parent_indices = tf.cast(
            tf.convert_to_tensor(value.parent_indices), tf.int32
        )
        cond = self.conditional_log_probs()
        n = self._n
        node_count = self._node_count

        batch_shape = ps.shape(parent_indices)[:-1]
        flat_parent_indices = tf.reshape(parent_indices, [-1, node_count - 1])

        if self._use_native:
            native = _native_conditional_clade()
            flat_log_prob = native.native_topology_log_prob(
                cond,
                flat_parent_indices,
                self._flat_parent,
                self._flat_child1,
                n,
            )
            return tf.reshape(flat_log_prob, batch_shape)

        table = self._flat_index_table
        pow2n = self._pow2n

        def single(pi):
            return tensor_ops.topology_log_prob(
                cond, pi, n, node_count, table, pow2n
            )

        flat_log_prob = tf.map_fn(
            single, flat_parent_indices, fn_output_signature=self.float_dtype
        )
        return tf.reshape(flat_log_prob, batch_shape)


__all__ = ["ConditionalCladeTreeDistribution"]
