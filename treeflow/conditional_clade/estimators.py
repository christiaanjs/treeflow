"""Gradient estimators for conditional clade topology distributions.

Sampling a topology is a discrete operation, so the gradient of an expectation
``E_{T ~ q_theta}[f(T)]`` with respect to the logits ``theta`` cannot be taken by
ordinary backpropagation. This module collects the estimators compared in the
accompanying notebook:

* :func:`straight_through_categorical` -- the "1/0 probability gradient" sampler.
  The forward pass returns a hard one-hot (literally ones and zeros); the
  backward pass pretends the sample was the softmax probability vector. A custom
  C++ op could implement exactly this forward/backward split; here it is done
  with the standard ``stop_gradient`` identity.
* :func:`gumbel_softmax` -- the Gumbel-Softmax / Concrete relaxation, with an
  optional straight-through (``hard=True``) variant.
* :func:`score_function_surrogate` -- REINFORCE, with an optional baseline.
* :func:`leave_one_out_baseline` -- the multi-sample leave-one-out control
  variate (the variance-reduction idea VIMCO is built on).
* :func:`vimco_surrogate` -- VIMCO, for the multi-sample importance-weighted
  bound.
* :func:`sample_relaxed_cost` -- the *reference* relaxation-based cost: draws a
  whole topology with straight-through / Gumbel-Softmax choices at every internal
  node by Python recursion. Correct but slow (a host-device sync per node); for
  training prefer the vectorised
  :func:`treeflow.conditional_clade.traversal_estimators.straight_through_traversal_cost`,
  which computes the same relaxed gradient on a pre-sampled traversal with no
  per-node Python.

The *surrogate* functions return a scalar whose ``tf.GradientTape`` gradient is
the desired estimator, so a training step is just ``tape.gradient(surrogate,
theta)``.

All estimators consume a *pre-sampled traversal* (the flat subsplit indices of a
batch of sampled topologies). The shared helpers that turn that traversal into
the differentiable per-sample quantities live in
:mod:`treeflow.conditional_clade.traversal_estimators` and are re-exported here.
"""

from __future__ import annotations

import typing as tp

import numpy as np
import tensorflow as tf

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.conditional_clade.clade import is_singleton
from treeflow.conditional_clade.distribution import ConditionalCladeDistribution
from treeflow.conditional_clade.traversal_estimators import (
    traversal_log_prob,
    straight_through_traversal_log_prob,
    straight_through_traversal_cost,
)


# ----------------------------------------------------------------------
# Discrete samplers with surrogate gradients
# ----------------------------------------------------------------------
def straight_through_categorical(
    logits: tf.Tensor, seed=None
) -> tp.Tuple[tf.Tensor, tf.Tensor]:
    """Sample a categorical one-hot with a straight-through gradient.

    Forward: a hard one-hot drawn from ``softmax(logits)`` (values in {0, 1}).
    Backward: the gradient is that of ``softmax(logits)`` -- the "1/0 forward,
    probability gradient" behaviour. Returns ``(one_hot, index)``.
    """
    probs = tf.nn.softmax(logits, axis=-1)
    index = tf.random.categorical(
        tf.math.log(probs)[tf.newaxis, :], 1, seed=seed
    )[0, 0]
    one_hot = tf.one_hot(index, tf.shape(logits)[-1], dtype=logits.dtype)
    # Forward value equals one_hot; gradient flows as through probs.
    straight_through = probs + tf.stop_gradient(one_hot - probs)
    return straight_through, index


def gumbel_softmax(
    logits: tf.Tensor, temperature: float = 0.5, hard: bool = True, seed=None
) -> tp.Tuple[tf.Tensor, tf.Tensor]:
    """Gumbel-Softmax / Concrete sample over a categorical.

    With ``hard=True`` the forward value is a one-hot (straight-through Gumbel-
    Softmax) while the gradient flows through the soft relaxation; with
    ``hard=False`` the soft relaxed sample is returned directly. Returns
    ``(sample, hard_index)``.
    """
    uniform = tf.random.uniform(
        tf.shape(logits), minval=1e-20, maxval=1.0, dtype=logits.dtype, seed=seed
    )
    gumbel = -tf.math.log(-tf.math.log(uniform))
    soft = tf.nn.softmax((logits + gumbel) / temperature, axis=-1)
    index = tf.argmax(soft, axis=-1)
    if not hard:
        return soft, index
    one_hot = tf.one_hot(index, tf.shape(logits)[-1], dtype=logits.dtype)
    straight_through = soft + tf.stop_gradient(one_hot - soft)
    return straight_through, index


# ----------------------------------------------------------------------
# Score-function (REINFORCE) family
# ----------------------------------------------------------------------
def score_function_surrogate(
    cost: tf.Tensor,
    log_prob: tf.Tensor,
    baseline: tp.Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """REINFORCE surrogate for ``minimising`` ``E_q[cost]``.

    ``cost`` and ``log_prob`` are per-sample tensors of equal shape; ``cost`` may
    itself depend on ``theta`` (e.g. ``cost = log q - log p``), in which case its
    pathwise gradient is included. The returned scalar's gradient is

        ``mean_i  (cost_i - baseline_i) grad log q(T_i)  +  mean_i grad cost_i``

    an unbiased estimate of ``grad E_q[cost]``. Minimise the returned scalar.
    """
    if baseline is None:
        baseline = tf.zeros_like(cost)
    advantage = tf.stop_gradient(cost - baseline)
    return tf.reduce_mean(advantage * log_prob + cost)


def leave_one_out_baseline(values: tf.Tensor) -> tf.Tensor:
    """Per-sample leave-one-out mean baseline.

    ``baseline_i = (sum_j values_j - values_i) / (K - 1)``. Used as a low-variance,
    sample-dependent baseline for :func:`score_function_surrogate` (the
    multi-sample control variate underpinning VIMCO). Requires ``K >= 2``.
    """
    values = tf.stop_gradient(values)
    k = tf.cast(tf.shape(values)[0], values.dtype)
    total = tf.reduce_sum(values)
    return (total - values) / (k - 1.0)


# ----------------------------------------------------------------------
# VIMCO (multi-sample importance-weighted bound)
# ----------------------------------------------------------------------
def _log_mean_exp(log_values: tf.Tensor) -> tf.Tensor:
    k = tf.cast(tf.shape(log_values)[0], log_values.dtype)
    return tf.reduce_logsumexp(log_values) - tf.math.log(k)


def vimco_surrogate(log_q: tf.Tensor, log_weights: tf.Tensor) -> tf.Tensor:
    """VIMCO surrogate for ``maximising`` the multi-sample bound.

    Maximises ``L = log (1/K sum_k exp(log_weights_k))`` where
    ``log_weights_k = log p(T_k) - log q(T_k)`` (with ``log_q`` differentiable in
    ``theta`` and ``log p`` treated as a constant). For a normalised target ``p``
    this bound is at most ``0`` and is tight when ``q = p``, so ascending it drives
    ``q`` toward ``p``.

    Uses the VIMCO per-sample control variate: the ``k``-th score term is weighted
    by the change in the bound when sample ``k``'s weight is replaced by the
    arithmetic mean of the others. Maximise the returned scalar.
    """
    log_weights = tf.convert_to_tensor(log_weights)
    k = tf.shape(log_weights)[0]
    k_float = tf.cast(k, log_weights.dtype)

    l_hat = _log_mean_exp(log_weights)  # the bound itself

    # Leave-one-out replacement: swap log_weights_k for the mean of the others.
    total = tf.reduce_sum(log_weights)
    loo_mean = (total - log_weights) / (k_float - 1.0)
    # Build, for each k, the vector of log weights with entry k replaced by
    # loo_mean_k, then its log-mean-exp.
    eye = tf.eye(k, dtype=log_weights.dtype)
    # row i = log_weights with position i replaced by loo_mean_i
    replaced = (
        log_weights[tf.newaxis, :] * (1.0 - eye) + loo_mean[:, tf.newaxis] * eye
    )
    l_hat_minus = tf.reduce_logsumexp(replaced, axis=-1) - tf.math.log(k_float)
    local_baseline = tf.stop_gradient(l_hat - l_hat_minus)

    normalised_weights = tf.stop_gradient(tf.nn.softmax(log_weights))

    score_term = tf.reduce_sum(local_baseline * log_q)
    pathwise_term = tf.reduce_sum(normalised_weights * log_weights)
    return score_term + pathwise_term


# ----------------------------------------------------------------------
# Relaxed recursive sampling (straight-through over the whole topology)
# ----------------------------------------------------------------------
class RelaxedSample(tp.NamedTuple):
    parent_indices: np.ndarray
    log_q: tf.Tensor  # differentiable in q logits
    log_p: tf.Tensor  # constant (p logits treated as fixed)


def sample_relaxed_cost(
    q_dist: ConditionalCladeDistribution,
    p_dist: tp.Optional[ConditionalCladeDistribution] = None,
    temperature: float = 0.5,
    gumbel: bool = True,
    seed=None,
) -> RelaxedSample:
    """Sample a whole topology with a relaxed gradient at every internal node.

    At each splittable clade reached from the root, a one-hot over that clade's
    subsplits is drawn with :func:`gumbel_softmax` (or
    :func:`straight_through_categorical` when ``gumbel=False``) using ``q_dist``'s
    logits. The hard choice determines which children to recurse into, so the
    realised topology is discrete, but ``log q`` (and ``log p``, if ``p_dist`` is
    given) accumulate through the relaxed one-hots and remain differentiable in
    ``q_dist``'s logits.

    The straight-through one-hots give the relaxation-based estimator: a single
    differentiable forward pass whose gradient estimates ``grad E_q[log q - log p]``
    (biased, but typically low variance).
    """
    support = q_dist.support
    q_cond = q_dist.conditional_log_probs()
    p_cond = p_dist.conditional_log_probs() if p_dist is not None else None
    if seed is not None:
        tf.random.set_seed(seed)

    assignment = {}
    log_q_terms = []
    log_p_terms = []

    def expand(clade: int) -> None:
        if is_singleton(clade):
            return
        parent_idx = support.parent_clade_index[clade]
        start = support.parent_offsets[parent_idx]
        subsplits = support.subsplits_by_parent[parent_idx]
        count = len(subsplits)
        logits_segment = q_dist.logits[start : start + count]
        if gumbel:
            one_hot, _ = gumbel_softmax(logits_segment, temperature, hard=True)
        else:
            one_hot, _ = straight_through_categorical(logits_segment)
        choice = int(tf.argmax(one_hot).numpy())
        q_segment = q_cond[start : start + count]
        log_q_terms.append(tf.reduce_sum(one_hot * q_segment))
        if p_cond is not None:
            p_segment = p_cond[start : start + count]
            log_p_terms.append(tf.reduce_sum(one_hot * p_segment))
        subsplit = subsplits[choice]
        assignment[clade] = subsplit
        expand(subsplit.child1)
        expand(subsplit.child2)

    expand(support.root_clade)
    parent_indices = support.assignment_to_parent_indices(assignment)
    log_q = tf.add_n(log_q_terms)
    log_p = (
        tf.add_n(log_p_terms)
        if log_p_terms
        else tf.constant(0.0, dtype=DEFAULT_FLOAT_DTYPE_TF)
    )
    return RelaxedSample(parent_indices=parent_indices, log_q=log_q, log_p=log_p)


__all__ = [
    "straight_through_categorical",
    "gumbel_softmax",
    "score_function_surrogate",
    "leave_one_out_baseline",
    "vimco_surrogate",
    "sample_relaxed_cost",
    "RelaxedSample",
    "traversal_log_prob",
    "straight_through_traversal_log_prob",
    "straight_through_traversal_cost",
]
