"""Gradient-estimator helpers over pre-sampled topology traversals.

A *traversal* is a batch of already-sampled topologies expressed as the flat
subsplit indices chosen at each internal node: an integer tensor
``flat_indices`` of shape ``[..., n-1]`` (one chosen subsplit per internal clade
on the path from the root). Sampling -- the discrete, data-dependent part -- is
done once up front (e.g. by
:meth:`ConditionalCladeDistribution.sample_flat_index_batch` or the native
sampler); the helpers here then compute the differentiable quantities every
gradient estimator needs from that fixed traversal, entirely with vectorised
TensorFlow graph ops.

This is the key to a fast straight-through estimator: the recursive
``sample_relaxed_cost`` walks the tree node by node in Python and forces a
host-device sync (``argmax(...).numpy()``) at every internal node to decide which
child to expand. Once a topology has been sampled, the choices are fixed, so the
relaxed gradient can be computed for the whole batch at once with no Python
recursion and no per-node syncs.

* :func:`traversal_log_prob` -- exact ``log q(T)`` (gather + sum). Its gradient is
  the score-function gradient used by REINFORCE / leave-one-out / VIMCO.
* :func:`straight_through_traversal_log_prob` -- a log-prob whose forward value is
  the exact ``log q(T)`` but whose gradient flows through the per-clade softmax
  (optionally Gumbel-Softmax), giving the low-variance biased straight-through
  estimator.
* :func:`straight_through_traversal_cost` -- the reverse-KL integrand
  ``log q(T) - log p(T)`` under a single shared relaxation, the convenient entry
  point for the relaxation-based estimator.
"""

from __future__ import annotations

import tensorflow as tf


def traversal_log_prob(
    conditional_log_probs: tf.Tensor, flat_indices: tf.Tensor
) -> tf.Tensor:
    """Exact ``log q(T)`` for pre-sampled topologies.

    Sums the conditional log-probabilities of the chosen subsplits over the last
    axis. ``flat_indices`` has shape ``[..., n-1]``; the result has shape
    ``[...]``. Differentiable in ``conditional_log_probs`` (the score-function
    gradient).
    """
    cond = tf.convert_to_tensor(conditional_log_probs)
    flat = tf.cast(flat_indices, tf.int32)
    return tf.reduce_sum(tf.gather(cond, flat), axis=-1)


def _relaxation_weights(
    relaxation_logits: tf.Tensor,
    decisions: tf.Tensor,
    segment_ids: tf.Tensor,
    temperature: float,
    gumbel: bool,
    seed,
) -> tf.Tensor:
    """Per-decision relaxed one-hot weights over the flat subsplit space.

    For each decision (an internal node), builds a softmax restricted to its
    parent clade's subsplit segment, optionally with Gumbel noise and a
    temperature. Returns a ``[D, M]`` tensor (``D`` decisions, ``M`` subsplits)
    that is zero outside each decision's segment and sums to one within it.
    """
    logits = tf.convert_to_tensor(relaxation_logits)
    seg_ids = tf.cast(segment_ids, tf.int32)
    seg_of_decision = tf.gather(seg_ids, decisions)  # [D]
    # Mask selecting, for each decision, the flat entries of its segment.
    mask = tf.equal(seg_ids[tf.newaxis, :], seg_of_decision[:, tf.newaxis])  # [D, M]
    scaled = tf.broadcast_to(logits[tf.newaxis, :], tf.shape(mask))
    if gumbel:
        if seed is None:
            seed = tf.random.uniform([2], maxval=2 ** 30, dtype=tf.int32)
        uniform = tf.random.stateless_uniform(
            tf.shape(mask),
            seed=tf.cast(seed, tf.int32),
            minval=1e-20,
            maxval=1.0,
            dtype=logits.dtype,
        )
        scaled = scaled + (-tf.math.log(-tf.math.log(uniform)))
    scaled = scaled / tf.cast(temperature, logits.dtype)
    neg_inf = tf.constant(-1e30, logits.dtype)
    relax_logits = tf.where(mask, scaled, neg_inf)
    return tf.nn.softmax(relax_logits, axis=-1)  # [D, M]


def _straight_through_apply(
    weights: tf.Tensor,
    value_log_probs: tf.Tensor,
    decisions: tf.Tensor,
    batch_shape: tf.Tensor,
) -> tf.Tensor:
    """Combine relaxed weights with the (possibly different) value log-probs.

    The forward value is the exact sum of the selected ``value_log_probs`` (the
    hard sample); the gradient flows through ``<weights, value_log_probs>`` --
    the straight-through trick.
    """
    cond = tf.convert_to_tensor(value_log_probs)
    hard = tf.gather(cond, decisions)  # [D]
    soft = tf.reduce_sum(weights * cond[tf.newaxis, :], axis=-1)  # [D]
    per_node = tf.stop_gradient(hard) + soft - tf.stop_gradient(soft)
    return tf.reduce_sum(tf.reshape(per_node, batch_shape), axis=-1)


def straight_through_traversal_log_prob(
    value_log_probs: tf.Tensor,
    relaxation_logits: tf.Tensor,
    flat_indices: tf.Tensor,
    segment_ids: tf.Tensor,
    temperature: float = 1.0,
    gumbel: bool = False,
    seed=None,
) -> tf.Tensor:
    """Straight-through ``log q(T)`` for pre-sampled topologies (vectorised).

    The forward value equals :func:`traversal_log_prob`; the gradient flows
    through the per-clade (Gumbel-)softmax relaxation built from
    ``relaxation_logits``. ``value_log_probs`` are the conditional
    log-probabilities whose selected entries are summed (and which the relaxation
    weights are dotted with); ``relaxation_logits`` are the raw per-subsplit
    logits defining the relaxation. With ``temperature == 1`` and ``gumbel ==
    False`` this is the plain straight-through ("1/0 probability gradient")
    estimator.
    """
    flat = tf.cast(flat_indices, tf.int32)
    decisions = tf.reshape(flat, [-1])
    weights = _relaxation_weights(
        relaxation_logits, decisions, segment_ids, temperature, gumbel, seed
    )
    return _straight_through_apply(weights, value_log_probs, decisions, tf.shape(flat))


def straight_through_traversal_cost(
    q_log_probs: tf.Tensor,
    p_log_probs: tf.Tensor,
    relaxation_logits: tf.Tensor,
    flat_indices: tf.Tensor,
    segment_ids: tf.Tensor,
    temperature: float = 1.0,
    gumbel: bool = False,
    seed=None,
) -> tf.Tensor:
    """Reverse-KL integrand ``log q(T) - log p(T)`` under one shared relaxation.

    Builds the relaxed weights once (from the variational ``relaxation_logits``)
    and applies them to both the variational ``q_log_probs`` and the target
    ``p_log_probs``, so the gradient flows through the relaxation for both terms,
    matching the recursive ``sample_relaxed_cost`` reference. Returns the
    per-sample cost of shape ``[...]`` (forward value equals the exact
    ``log q - log p`` of the sampled trees).
    """
    flat = tf.cast(flat_indices, tf.int32)
    decisions = tf.reshape(flat, [-1])
    batch_shape = tf.shape(flat)
    weights = _relaxation_weights(
        relaxation_logits, decisions, segment_ids, temperature, gumbel, seed
    )
    log_q = _straight_through_apply(weights, q_log_probs, decisions, batch_shape)
    log_p = _straight_through_apply(weights, p_log_probs, decisions, batch_shape)
    return log_q - log_p


__all__ = [
    "traversal_log_prob",
    "straight_through_traversal_log_prob",
    "straight_through_traversal_cost",
]
