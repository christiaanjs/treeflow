"""Rooted-tree Jukes-Cantor phylogenetic likelihood for VBPI / topology MCMC.

Everything in the VBPI stack works on *rooted* binary topologies, and so does
this likelihood: Felsenstein's pruning runs postorder to the root and contracts
the root partials against the equilibrium frequencies. For the reversible
Jukes-Cantor model that root contraction is the exact tree likelihood regardless
of rooting, so no unrooted conversion is needed.

It reuses treeflow's existing machinery rather than reimplementing it:

* transition matrices come from the :class:`~treeflow.evolution.substitution.nucleotide.jc.JC`
  substitution model's eigendecomposition
  (:func:`~treeflow.evolution.substitution.probabilities.get_transition_probabilities_eigen`);
* the pruning is the **native** C++ likelihood op
  (:func:`~treeflow.acceleration.native.native_phylogenetic_log_likelihood_rescaled`)
  when it is built, falling back to treeflow's pure-TensorFlow reference.

The scoring is differentiable in the branch lengths (used by VBPI's
reparameterised branch gradients) and, wrapped in ``tf.function`` via
:func:`make_jc_log_likelihood_fn`, fast enough to score one tree per step in the
topology MCMC.
"""
import typing as tp

import numpy as np
import tensorflow as tf

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.evolution.substitution.nucleotide.jc import JC
from treeflow.evolution.substitution.probabilities import (
    get_transition_probabilities_eigen,
)
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
from treeflow.tree.topology.tensorflow_tree_topology import (
    TensorflowTreeTopology,
    numpy_topology_to_tensor,
)

def _jc_eigen(dtype: tf.DType):
    # Not cached: the eigendecomposition is a handful of tf.constants, and
    # caching them would capture tensors created inside whichever tf.function
    # first built them, making them unusable ("out of scope") from a later graph.
    return JC().eigen(JC.frequencies(dtype=dtype), dtype=dtype)


def jc_frequencies(dtype: tf.DType = DEFAULT_FLOAT_DTYPE_TF) -> tf.Tensor:
    return JC.frequencies(dtype=dtype)


def jc_transition_probs(branch_lengths: tf.Tensor) -> tf.Tensor:
    """Jukes-Cantor transition matrices ``P(t)`` from the JC eigendecomposition.

    ``branch_lengths`` has shape ``[...]``; the result is ``[..., 4, 4]``.
    Differentiable in ``branch_lengths``.
    """
    branch_lengths = tf.convert_to_tensor(branch_lengths)
    return get_transition_probabilities_eigen(
        _jc_eigen(branch_lengths.dtype), branch_lengths
    )


def _log_likelihood_from_transition_probs(
    topology: TensorflowTreeTopology,
    sequences_onehot: tf.Tensor,
    transition_probs: tf.Tensor,
    frequencies: tf.Tensor,
    use_native: bool,
) -> tf.Tensor:
    """Per-site log likelihood via the native op (or the TF reference)."""
    if use_native:
        from treeflow.acceleration.native import (
            native_phylogenetic_log_likelihood_rescaled,
        )

        return native_phylogenetic_log_likelihood_rescaled(
            sequences_onehot,
            transition_probs,
            frequencies,
            topology.postorder_node_indices,
            topology.node_child_indices,
        )
    from treeflow.traversal.phylo_likelihood import (
        phylogenetic_log_likelihood_rescaled,
    )

    # The reference traversal ``tf.stack``s sibling partials without broadcasting,
    # so (unlike the native op) the leaf partials must already carry the full
    # sample batch. ``transition_probs`` is [*B, 1, node, 4, 4]; broadcast the
    # sequences to [*B, n_sites, n_leaf, 4].
    sample_batch = tf.shape(transition_probs)[:-4]  # [*B]
    sequences_onehot = tf.broadcast_to(
        sequences_onehot,
        tf.concat([sample_batch, tf.shape(sequences_onehot)], axis=0),
    )
    batch_shape = tf.broadcast_dynamic_shape(
        tf.shape(sequences_onehot)[:-2], tf.shape(transition_probs)[:-3]
    )
    return phylogenetic_log_likelihood_rescaled(
        topology, sequences_onehot, transition_probs, frequencies, batch_shape
    )


def _native_available(use_native: tp.Union[bool, str]) -> bool:
    if use_native == "auto":
        try:
            from treeflow.acceleration.native import is_available

            return is_available()
        except Exception:
            return False
    if isinstance(use_native, bool):
        return use_native
    raise ValueError(f"use_native must be True, False or 'auto'; got {use_native!r}")


def _transition_probs_with_root(branch_lengths: tf.Tensor) -> tf.Tensor:
    """Per-node transition matrices ``[*B, 1, 2n-1, 4, 4]`` (root slot = identity).

    The root (last node id) is never read as a child, so its matrix is arbitrary.
    A length-1 "sites" axis is inserted so the native/reference batch broadcasts
    the per-branch matrices against the per-site sequences (sites is the sequence
    batch, so the result batch is ``[*B, n_sites]``).
    """
    branch_probs = jc_transition_probs(branch_lengths)  # [*B, 2n-2, 4, 4]
    sample_shape = tf.shape(branch_lengths)[:-1]
    eye = tf.eye(4, dtype=branch_probs.dtype)
    root_probs = tf.broadcast_to(eye, tf.concat([sample_shape, [1, 4, 4]], axis=0))
    transition_probs = tf.concat([branch_probs, root_probs], axis=-3)  # [*B, 2n-1,4,4]
    return tf.expand_dims(transition_probs, axis=-4)  # [*B, 1, 2n-1, 4, 4]


def rooted_jc_log_likelihood(
    parent_indices: np.ndarray,
    leaf_partials: tf.Tensor,
    branch_lengths: tf.Tensor,
    topology: tp.Optional[TensorflowTreeTopology] = None,
    use_native: tp.Union[bool, str] = "auto",
) -> tf.Tensor:
    """Log likelihood ``log P(D | T, b)`` of a single rooted tree under JC.

    Parameters
    ----------
    parent_indices
        Rooted topology, length ``2n-2`` (ignored if ``topology`` given).
    leaf_partials
        Leaf state partials ``[n_leaves, n_sites, 4]`` ordered by taxon id.
    branch_lengths
        Non-root branch lengths ``[..., 2n-2]`` indexed by node id. Differentiable.
    topology
        Optional pre-built ``TensorflowTreeTopology`` (avoids rebuilding the index
        arrays when scoring many branch-length samples on one tree).
    use_native
        ``"auto"`` (default) uses the native op when built, else the reference.

    Returns
    -------
    Scalar (or ``[...]``) log likelihood summed over sites.
    """
    branch_lengths = tf.convert_to_tensor(branch_lengths, dtype=DEFAULT_FLOAT_DTYPE_TF)
    if topology is None:
        topology = numpy_topology_to_tensor(
            NumpyTreeTopology(parent_indices=np.asarray(parent_indices))
        )
    leaf_partials = tf.convert_to_tensor(leaf_partials, dtype=branch_lengths.dtype)
    sequences_onehot = tf.transpose(leaf_partials, [1, 0, 2])  # [n_sites, n_leaf, 4]

    transition_probs = _transition_probs_with_root(branch_lengths)
    site_log_likelihood = _log_likelihood_from_transition_probs(
        topology,
        sequences_onehot,
        transition_probs,
        jc_frequencies(branch_lengths.dtype),
        _native_available(use_native),
    )
    return tf.reduce_sum(site_log_likelihood, axis=-1)


def make_jc_log_likelihood_fn(
    leaf_partials: tf.Tensor,
    use_native: tp.Union[bool, str] = "auto",
) -> tp.Callable[[TensorflowTreeTopology, tf.Tensor], tf.Tensor]:
    """A ``tf.function``-compiled ``(topology, branch_lengths) -> log lik`` closure.

    The alignment is captured once. The returned function is traced once per
    ``(taxon_count, dtype)`` and then called with each proposed topology's index
    tensors and branch lengths, so the topology MCMC pays the graph-build cost
    only once -- the "function mode" fast path.
    """
    leaf_partials = tf.convert_to_tensor(leaf_partials, dtype=DEFAULT_FLOAT_DTYPE_TF)
    sequences_onehot = tf.transpose(leaf_partials, [1, 0, 2])
    frequencies = jc_frequencies(DEFAULT_FLOAT_DTYPE_TF)
    native = _native_available(use_native)

    @tf.function
    def log_likelihood(topology: TensorflowTreeTopology, branch_lengths: tf.Tensor):
        transition_probs = _transition_probs_with_root(branch_lengths)
        site_ll = _log_likelihood_from_transition_probs(
            topology, sequences_onehot, transition_probs, frequencies, native
        )
        return tf.reduce_sum(site_ll, axis=-1)

    return log_likelihood


__all__ = [
    "jc_frequencies",
    "jc_transition_probs",
    "rooted_jc_log_likelihood",
    "make_jc_log_likelihood_fn",
]
