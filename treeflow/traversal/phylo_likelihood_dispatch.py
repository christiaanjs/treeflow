"""Intelligent dispatch between the rescaled and unrescaled likelihoods.

Rescaling (dividing partial likelihoods by their per-site maximum at every
internal node and accumulating the log scale factors) is the standard way to
keep the phylogenetic likelihood numerically stable on large/deep trees, where
the unscaled partials underflow to zero. It costs a little extra work per node,
so for small trees the unscaled version is both fine and faster.

:func:`phylogenetic_log_likelihood` returns the per-site ``log`` likelihood and
chooses between the two based on the ``rescaling`` argument:

* ``False``  -- never rescale (fastest; may underflow on large trees);
* ``True``   -- always rescale (most stable);
* ``"auto"`` -- pick statically from the leaf count and floating-point type,
  using :func:`default_rescaling_threshold` (the default; no runtime overhead);
* ``"adaptive"`` -- compute the unscaled likelihood and, only if it is not
  finite, fall back to the rescaled one (data-driven, via ``tf.cond``).

Both the pure-TensorFlow and the native C++ implementations are supported via
``use_native``.
"""
import numpy as np
import tensorflow as tf

from treeflow.traversal.phylo_likelihood import (
    phylogenetic_likelihood,
    phylogenetic_log_likelihood_rescaled,
)


def default_rescaling_threshold(
    dtype, safety: float = 0.6, orders_lost_per_taxon: float = 1.0
) -> int:
    """Leaf count above which ``"auto"`` switches to the rescaled likelihood.

    Heuristic: a per-site root partial loses roughly ``orders_lost_per_taxon``
    decimal orders of magnitude per added taxon (data dependent; ~1 is a
    reasonable default for nucleotide data), and the smallest representable
    positive normal of ``dtype`` gives the underflow budget in orders of
    magnitude. We switch to rescaling at ``safety`` of that budget so we stay
    comfortably clear of underflow.

    For the defaults this gives ~184 taxa for float64 and ~22 for float32.
    """
    np_dtype = tf.as_dtype(dtype).as_numpy_dtype
    underflow_budget = -np.log10(np.finfo(np_dtype).tiny)
    return max(1, int(safety * underflow_budget / orders_lost_per_taxon))


def _static_leaf_count(sequences_onehot, postorder_node_indices):
    """Best-effort static leaf count; None if shapes are unknown."""
    leaf_dim = sequences_onehot.shape[-2]
    if leaf_dim is not None:
        return int(leaf_dim)
    postorder_dim = postorder_node_indices.shape[0]
    if postorder_dim is not None:
        return int(postorder_dim) + 1  # internal nodes = leaves - 1
    return None


def phylogenetic_log_likelihood(
    topology,
    sequences_onehot: tf.Tensor,
    transition_probs: tf.Tensor,
    frequencies: tf.Tensor,
    batch_shape=(),
    use_native: bool = False,
    rescaling="auto",
    rescaling_threshold: int = None,
    block_size: int = 1,
    unroll="auto",
) -> tf.Tensor:
    """Per-site phylogenetic LOG likelihood, dispatching rescaled/unrescaled.

    Parameters match :func:`treeflow.traversal.phylo_likelihood.phylogenetic_likelihood`
    plus:

    use_native
        If True, use the native C++ ops; otherwise the pure-TensorFlow ones.
    rescaling
        ``False``, ``True``, ``"auto"`` (default) or ``"adaptive"`` -- see the
        module docstring.
    rescaling_threshold
        Override the leaf-count threshold used by ``"auto"``.
    block_size
        Site-blocking width for the native ops (SIMD); ignored when
        ``use_native`` is False. See
        :func:`treeflow.acceleration.native.native_phylogenetic_likelihood`.
    """
    if use_native:
        from treeflow.acceleration.native import (
            native_phylogenetic_likelihood,
            native_phylogenetic_log_likelihood_rescaled,
        )

        native_args = (
            sequences_onehot,
            transition_probs,
            frequencies,
            topology.postorder_node_indices,
            topology.node_child_indices,
        )

        def unscaled_log():
            return tf.math.log(
                native_phylogenetic_likelihood(
                    *native_args, batch_shape=batch_shape, block_size=block_size
                )
            )

        def rescaled_log():
            return native_phylogenetic_log_likelihood_rescaled(
                *native_args, batch_shape=batch_shape, block_size=block_size
            )

    else:
        args = (topology, sequences_onehot, transition_probs, frequencies)

        def unscaled_log():
            return tf.math.log(
                phylogenetic_likelihood(*args, batch_shape=batch_shape, unroll=unroll)
            )

        def rescaled_log():
            return phylogenetic_log_likelihood_rescaled(
                *args, batch_shape=batch_shape, unroll=unroll
            )

    if rescaling is True:
        return rescaled_log()
    if rescaling is False:
        return unscaled_log()
    if rescaling == "auto":
        leaf_count = _static_leaf_count(
            sequences_onehot, topology.postorder_node_indices
        )
        threshold = (
            rescaling_threshold
            if rescaling_threshold is not None
            else default_rescaling_threshold(frequencies.dtype)
        )
        if leaf_count is None or leaf_count >= threshold:
            return rescaled_log()
        return unscaled_log()
    if rescaling == "adaptive":
        value = unscaled_log()
        all_finite = tf.reduce_all(tf.math.is_finite(value))
        return tf.cond(all_finite, lambda: value, rescaled_log)
    raise ValueError(
        f"Unknown rescaling mode {rescaling!r}; expected False, True, "
        "'auto' or 'adaptive'."
    )


__all__ = [
    "phylogenetic_log_likelihood",
    "default_rescaling_threshold",
]
