"""Pure-TensorFlow, graph-compatible operations for conditional clade topologies.

The eager sampler in :mod:`treeflow.conditional_clade.distribution` uses Python
recursion, NumPy random numbers and ``.numpy()`` calls, so it cannot run inside a
``tf.function``. The functions here re-implement sampling and log-probability
entirely with tensors and ``tf.while_loop`` / ``tf.cond``, so they trace into a
graph (and work under ``tf.function``).

The recursive, data-dependent structure of a topology is handled with an
explicit two-state depth-first traversal driven by a ``tf.while_loop``:

* a clade is first *entered* -- a subsplit is sampled and its two children are
  pushed onto the stack, followed by the clade itself marked for *leaving*;
* when the clade is *left*, all of its descendants have already been processed,
  so it is assigned the next internal-node index.

Because a clade is assigned its index only on the way out, internal nodes are
numbered in post-order, giving the labelling TreeFlow expects (leaves ``0..n-1``,
internal ``n..2n-2``, root last, every parent index larger than its children).

Clades are integer bitsets, so per-clade lookups (segment offsets, sampled
children, assigned node ids) are dense tensors of length ``2**n`` -- fine for the
small taxon sets this exact machinery targets.
"""

from __future__ import annotations

import typing as tp

import numpy as np
import tensorflow as tf

INT = tf.int32


def _scatter1(tensor: tf.Tensor, index: tf.Tensor, value: tf.Tensor) -> tf.Tensor:
    """Functional single-element update ``tensor[index] = value``."""
    return tf.tensor_scatter_nd_update(tensor, [[index]], [value])


def sample_parent_indices(
    logits: tf.Tensor,
    node_id_init: tf.Tensor,
    clade_offset: tf.Tensor,
    clade_count: tf.Tensor,
    flat_child1: tf.Tensor,
    flat_child2: tf.Tensor,
    taxon_count: int,
    node_count: int,
    seed: tf.Tensor,
) -> tf.Tensor:
    """Sample a topology as ``parent_indices`` (length ``2n-2``) in graph mode.

    ``seed`` is a length-2 stateless seed tensor. All other arguments are
    constants derived from a :class:`ConditionalCladeSupport`:

    * ``node_id_init`` -- length ``2**n``; leaf clades pre-filled with their taxon
      index, everything else ``-1``.
    * ``clade_offset`` / ``clade_count`` -- length ``2**n``; per-parent-clade start
      offset and number of subsplits in the flat logit layout.
    * ``flat_child1`` / ``flat_child2`` -- length ``M``; child bitsets of each flat
      subsplit.
    """
    pow2n = 1 << taxon_count
    capacity = 2 * node_count + 2
    root_clade = pow2n - 1

    stack_clade = tf.tensor_scatter_nd_update(
        tf.zeros([capacity], INT), [[0]], [root_clade]
    )
    stack_state = tf.zeros([capacity], INT)
    node_id = tf.identity(node_id_init)
    parent_indices = tf.fill([node_count - 1], tf.constant(-1, INT))
    child1_of = tf.fill([pow2n], tf.constant(-1, INT))
    child2_of = tf.fill([pow2n], tf.constant(-1, INT))
    top = tf.constant(1, INT)
    next_id = tf.constant(taxon_count, INT)

    def cond(top, *_):
        return top > 0

    def body(
        top, stack_clade, stack_state, node_id, next_id, parent_indices,
        child1_of, child2_of, seed,
    ):
        top = top - 1
        clade = stack_clade[top]
        state = stack_state[top]

        def leave():
            new_id = next_id
            nid = _scatter1(node_id, clade, new_id)
            c1 = child1_of[clade]
            c2 = child2_of[clade]
            id1 = nid[c1]
            id2 = nid[c2]
            pidx = tf.tensor_scatter_nd_update(
                parent_indices, [[id1], [id2]], [new_id, new_id]
            )
            return (
                top, stack_clade, stack_state, nid, next_id + 1, pidx,
                child1_of, child2_of, seed,
            )

        def enter():
            is_single = tf.logical_and(
                clade > 0,
                tf.equal(tf.bitwise.bitwise_and(clade, clade - 1), 0),
            )

            def leaf():
                return (
                    top, stack_clade, stack_state, node_id, next_id,
                    parent_indices, child1_of, child2_of, seed,
                )

            def split():
                offset = clade_offset[clade]
                count = clade_count[clade]
                seg = tf.slice(logits, [offset], [count])
                parts = tf.random.experimental.stateless_split(seed, 2)
                local = tf.cast(
                    tf.random.stateless_categorical(seg[tf.newaxis], 1, parts[0])[0, 0],
                    INT,
                )
                flat = offset + local
                cc1 = flat_child1[flat]
                cc2 = flat_child2[flat]
                c1_of = _scatter1(child1_of, clade, cc1)
                c2_of = _scatter1(child2_of, clade, cc2)
                # Push self (to be left later), then both children to enter.
                t = top
                sc = _scatter1(stack_clade, t, clade)
                ss = _scatter1(stack_state, t, tf.constant(1, INT))
                t = t + 1
                sc = _scatter1(sc, t, cc2)
                ss = _scatter1(ss, t, tf.constant(0, INT))
                t = t + 1
                sc = _scatter1(sc, t, cc1)
                ss = _scatter1(ss, t, tf.constant(0, INT))
                t = t + 1
                return (
                    t, sc, ss, node_id, next_id, parent_indices,
                    c1_of, c2_of, parts[1],
                )

            return tuple(tf.cond(is_single, leaf, split))

        return tuple(tf.cond(tf.equal(state, 1), leave, enter))

    result = tf.while_loop(
        cond,
        body,
        (
            top, stack_clade, stack_state, node_id, next_id, parent_indices,
            child1_of, child2_of, seed,
        ),
        maximum_iterations=3 * node_count,
    )
    return result[5]


def parent_indices_to_child_indices(
    parent_indices: tf.Tensor, node_count: int
) -> tf.Tensor:
    """Derive ``child_indices`` (``[node_count, 2]``) from ``parent_indices``.

    Each internal node has exactly two children; the smaller-indexed child goes
    in column 0 (matching the NumPy convention). Leaf rows are ``-1``.
    """
    child_ids = tf.range(node_count - 1, dtype=INT)
    parents = tf.cast(parent_indices, INT)
    cmin = tf.math.unsorted_segment_min(child_ids, parents, node_count)
    cmax = tf.math.unsorted_segment_max(child_ids, parents, node_count)
    counts = tf.math.unsorted_segment_sum(
        tf.ones_like(child_ids), parents, node_count
    )
    has_children = counts > 0
    col0 = tf.where(has_children, cmin, -1)
    col1 = tf.where(has_children, cmax, -1)
    return tf.stack([col0, col1], axis=1)


def child_indices_to_preorder(
    child_indices: tf.Tensor, node_count: int
) -> tf.Tensor:
    """Pre-order traversal (length ``node_count``) from ``child_indices``."""
    root = node_count - 1
    stack = tf.tensor_scatter_nd_update(
        tf.zeros([node_count], INT), [[0]], [root]
    )
    out = tf.TensorArray(INT, size=node_count)
    top = tf.constant(1, INT)
    counter = tf.constant(0, INT)

    def cond(top, stack, out, counter):
        return top > 0

    def body(top, stack, out, counter):
        top = top - 1
        node = stack[top]
        out = out.write(counter, node)
        counter = counter + 1
        c0 = child_indices[node, 0]
        c1 = child_indices[node, 1]

        def push():
            t = top
            s = _scatter1(stack, t, c1)
            t = t + 1
            s = _scatter1(s, t, c0)
            t = t + 1
            return t, s

        new_top, new_stack = tf.cond(c0 >= 0, push, lambda: (top, stack))
        return (new_top, new_stack, out, counter)

    _, _, out, _ = tf.while_loop(
        cond, body, (top, stack, out, counter), maximum_iterations=node_count
    )
    return out.stack()


def _build_clade_of(
    child_indices: tf.Tensor, taxon_count: int, node_count: int
) -> tf.Tensor:
    """Bitset clade of every node, by an ascending (post-order) OR accumulation."""
    leaf_clades = tf.bitwise.left_shift(
        tf.ones([taxon_count], INT), tf.range(taxon_count, dtype=INT)
    )
    clade_of = tf.concat(
        [leaf_clades, tf.zeros([node_count - taxon_count], INT)], axis=0
    )
    i = tf.constant(taxon_count, INT)

    def cond(i, clade_of):
        return i < node_count

    def body(i, clade_of):
        c0 = child_indices[i, 0]
        c1 = child_indices[i, 1]
        value = tf.bitwise.bitwise_or(clade_of[c0], clade_of[c1])
        return (i + 1, _scatter1(clade_of, i, value))

    _, clade_of = tf.while_loop(
        cond, body, (i, clade_of), maximum_iterations=node_count
    )
    return clade_of


def topology_log_prob(
    conditional_log_probs: tf.Tensor,
    parent_indices: tf.Tensor,
    taxon_count: int,
    node_count: int,
    flat_index_table: tf.lookup.StaticHashTable,
    pow2n: int,
) -> tf.Tensor:
    """Log-probability of a single topology, fully in graph mode.

    Computes the clade at each node, reads off the subsplit at every internal
    node, maps it to its flat index via ``flat_index_table`` (keyed by
    ``parent * 2**n + canonical_child1``) and sums the conditional
    log-probabilities. Differentiable in ``conditional_log_probs``.
    """
    parent_indices = tf.cast(parent_indices, INT)
    child_indices = parent_indices_to_child_indices(parent_indices, node_count)
    clade_of = _build_clade_of(child_indices, taxon_count, node_count)

    internal_child = child_indices[taxon_count:]  # [n-1, 2]
    cl0 = tf.gather(clade_of, internal_child[:, 0])
    cl1 = tf.gather(clade_of, internal_child[:, 1])
    # Canonical child1 contains the smallest taxon: smaller lowest set bit.
    low0 = tf.bitwise.bitwise_and(cl0, tf.negative(cl0))
    low1 = tf.bitwise.bitwise_and(cl1, tf.negative(cl1))
    canonical_child1 = tf.where(low0 < low1, cl0, cl1)
    parent_clade = tf.bitwise.bitwise_or(cl0, cl1)
    keys = tf.cast(parent_clade, tf.int64) * tf.cast(pow2n, tf.int64) + tf.cast(
        canonical_child1, tf.int64
    )
    flat = flat_index_table.lookup(keys)
    return tf.reduce_sum(tf.gather(conditional_log_probs, flat))


__all__ = [
    "sample_parent_indices",
    "parent_indices_to_child_indices",
    "child_indices_to_preorder",
    "topology_log_prob",
]
