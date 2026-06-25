"""Connect the relaxed phylogenetic likelihood's gradient to the clade model.

:func:`clade_straight_through_log_likelihood` runs Felsenstein pruning over a
*sampled* topology, but at every internal node selects its children with a
straight-through softmax over the **clade model**'s subsplit probabilities, so
``d log L / d (conditional log-probs)`` flows back to the clade logits.

The forward pass is the exact likelihood of the sampled tree (the realised
subsplit is selected). The backward pass contrasts the realised subsplit against a
set of *candidate* subsplits at that clade -- and the informative signal lives in
that contrast, so the candidates' combined child-partials must exist. How those
alternative-child partials are produced is pluggable:

* ``candidate_subsplits_fn(clade, realised) -> [subsplits]`` chooses the candidate
  set (every subsplit for an exhaustive support; the realised one plus a sample
  for a restricted / embedding support);
* ``alternative_partial_fn(clade) -> partials`` supplies the partial likelihood of
  a *non-realised* child clade (e.g. by sampling a subtree from the clade model),
  with realised clades reusing the partials already computed in the tree.

Because :func:`treeflow.conditional_clade.relaxed_likelihood.straight_through_gather`
computes each candidate's gradient independently, a sampled candidate set just
yields a sampled-softmax approximation -- the routing itself stays exact, which is
what the tests check (gather-routing == dense one-hot multiply).

A single fixed transition matrix ``P`` (equal branch lengths) is used here to
isolate the *topology* gradient; per-edge branch lengths are an orthogonal
extension.
"""

from __future__ import annotations

import typing as tp

import tensorflow as tf

from treeflow.conditional_clade.clade import (
    Subsplit,
    enumerate_clade_subsplits,
    full_clade,
    is_singleton,
    min_taxon,
    popcount,
)
from treeflow.conditional_clade.relaxed_likelihood import straight_through_gather


def _apply_transition(transition: tf.Tensor, partials: tf.Tensor) -> tf.Tensor:
    """``(P . partial)_a = sum_b P[a, b] partial[..., b]`` -> ``[..., state]``."""
    return tf.linalg.matvec(transition, partials)


def _as_transition_fn(transition):
    """Normalise ``transition`` to a callable ``clade -> [state, state]`` matrix.

    Accepts either a fixed matrix (used on every edge -- equal branch lengths) or
    a callable giving the transition matrix on the edge above a given child clade
    (variable branch lengths, e.g. from a node-height ratio transform).
    """
    if callable(transition):
        return transition
    matrix = tf.convert_to_tensor(transition)
    return lambda clade: matrix


def exhaustive_candidates(clade: int, realised: Subsplit) -> tp.List[Subsplit]:
    """Every subsplit of ``clade`` (for an enumerable support)."""
    return enumerate_clade_subsplits(clade)


def sampled_candidates(
    n_alternatives: int, seed=None
) -> tp.Callable[[int, Subsplit], tp.List[Subsplit]]:
    """Candidate chooser: the realised subsplit plus ``n_alternatives`` sampled ones."""
    import numpy as np

    rng = np.random.default_rng(seed)

    def choose(clade: int, realised: Subsplit) -> tp.List[Subsplit]:
        options = enumerate_clade_subsplits(clade)
        if len(options) <= n_alternatives + 1:
            return options
        others = [s for s in options if s != realised]
        picks = rng.choice(len(others), size=n_alternatives, replace=False)
        return [realised] + [others[i] for i in picks]

    return choose


def sampled_subtree_partial_fn(
    q_distribution,
    leaf_partials: tp.Dict[int, tf.Tensor],
    transition: tf.Tensor,
    seed=None,
) -> tp.Callable[[int], tf.Tensor]:
    """Alternative-child partials by sampling a subtree from the clade model.

    Returns a cached function ``clade -> partials`` that, for a non-realised child
    clade, samples a subsplit resolution of that clade from ``q_distribution`` and
    prunes it (with the fixed transition matrix), giving a well-defined -- if
    approximate -- partial for the gradient contrast. Singleton clades return the
    leaf partial.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    support = q_distribution.support
    cond_probs = q_distribution.conditional_probs_numpy()
    transition_fn = _as_transition_fn(transition)
    cache: tp.Dict[int, tf.Tensor] = dict(leaf_partials)

    def partial(clade: int) -> tf.Tensor:
        if clade in cache:
            return cache[clade]
        if is_singleton(clade):
            result = leaf_partials[clade]
        else:
            parent_idx = support.parent_clade_index[clade]
            start = support.parent_offsets[parent_idx]
            subsplits = support.subsplits_by_parent[parent_idx]
            probs = cond_probs[start : start + len(subsplits)]
            probs = probs / probs.sum()
            chosen = subsplits[rng.choice(len(subsplits), p=probs)]
            left = _apply_transition(transition_fn(chosen.child1), partial(chosen.child1))
            right = _apply_transition(transition_fn(chosen.child2), partial(chosen.child2))
            result = left * right
        cache[clade] = result
        return result

    return partial


def clade_straight_through_log_likelihood(
    q_distribution,
    parent_indices,
    sequences_onehot: tf.Tensor,
    transition,
    frequencies: tf.Tensor,
    candidate_subsplits_fn: tp.Optional[
        tp.Callable[[int, Subsplit], tp.List[Subsplit]]
    ] = None,
    alternative_partial_fn: tp.Optional[tp.Callable[[int], tf.Tensor]] = None,
    gather: bool = True,
) -> tf.Tensor:
    """Per-site log-likelihood of a sampled tree, differentiable in the clade model.

    Parameters
    ----------
    q_distribution
        The clade model (a ``ConditionalCladeDistribution``); gradients flow to its
        ``conditional_log_probs``.
    parent_indices
        The sampled topology (TreeFlow ``parent_indices``).
    sequences_onehot
        ``[..., leaf, state]`` one-hot leaf sequences.
    transition
        ``[state, state]`` fixed transition matrix (equal branch lengths).
    frequencies
        ``[state]`` root frequencies.
    candidate_subsplits_fn, alternative_partial_fn
        See module docstring. Default to the exhaustive candidate set and
        sampled-subtree alternative partials.
    gather
        ``True`` (default): efficient straight-through gather. ``False``: dense
        one-hot multiply (the validation reference).

    Returns
    -------
    Per-site log-likelihood ``[...]``.
    """
    support = q_distribution.support
    n = support.taxon_count
    conditional_log_probs = q_distribution.conditional_log_probs()
    transition_fn = _as_transition_fn(transition)

    assignment = support.parent_indices_to_assignment(parent_indices)
    leaf_partials = {1 << i: sequences_onehot[..., i, :] for i in range(n)}

    if candidate_subsplits_fn is None:
        candidate_subsplits_fn = exhaustive_candidates
    if alternative_partial_fn is None:
        alternative_partial_fn = sampled_subtree_partial_fn(
            q_distribution, leaf_partials, transition
        )

    def child_partial(clade: int) -> tf.Tensor:
        if clade in realised_partials:
            return realised_partials[clade]
        if is_singleton(clade):
            return leaf_partials[clade]
        return alternative_partial_fn(clade)

    realised_partials: tp.Dict[int, tf.Tensor] = dict(leaf_partials)
    # Process realised splittable clades children-before-parents (increasing size).
    for clade in sorted(assignment.keys(), key=popcount):
        realised = assignment[clade]
        candidates = candidate_subsplits_fn(clade, realised)
        realised_position = candidates.index(realised)

        combined = []
        logits = []
        for subsplit in candidates:
            left = _apply_transition(
                transition_fn(subsplit.child1), child_partial(subsplit.child1)
            )
            right = _apply_transition(
                transition_fn(subsplit.child2), child_partial(subsplit.child2)
            )
            combined.append(left * right)
            flat = support.flat_index[(clade, subsplit)]
            logits.append(conditional_log_probs[flat])

        logits = tf.stack(logits)  # [num_candidates]
        soft = tf.nn.softmax(logits)
        one_hot = tf.one_hot(realised_position, len(candidates), dtype=soft.dtype)
        selection = one_hot + soft - tf.stop_gradient(soft)  # straight-through
        values = tf.stack(combined, axis=0)  # [num_candidates, ..., state]

        if gather:
            realised_partials[clade] = straight_through_gather(
                values, selection[tf.newaxis]
            )[0]
        else:
            realised_partials[clade] = tf.tensordot(
                selection[tf.newaxis], values, axes=[[1], [0]]
            )[0]

    root_partial = realised_partials[full_clade(n)]
    site_likelihood = tf.reduce_sum(frequencies * root_partial, axis=-1)
    return tf.math.log(site_likelihood)


__all__ = [
    "clade_straight_through_log_likelihood",
    "exhaustive_candidates",
    "sampled_candidates",
    "sampled_subtree_partial_fn",
]
