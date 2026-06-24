"""Native (C++) ops for conditional clade topology distributions.

Compiled drop-in replacements for the graph-mode TensorFlow reference in
:mod:`treeflow.conditional_clade.tensor_ops`:

* :func:`native_sample_parent_indices` -- sample topologies (``parent_indices``)
  from per-subsplit logits;
* :func:`native_topology_log_prob` -- log-probability of a topology, with an
  analytic (scatter-add) gradient w.r.t. the conditional log-probabilities;
* :func:`native_parent_indices_to_child_indices` and
  :func:`native_child_indices_to_preorder` -- the topology index transforms.

The recursive structure of a topology is expressed as ordinary C++ recursion
rather than a ``tf.while_loop``, so these are both simpler and faster than the
graph-mode reference, while producing identical results.

The ops are wired into TensorFlow the same way as the other native ops in this
package (``tf.load_op_library`` + ``tf.RegisterGradient``); the ``.so`` is built
on demand via :mod:`treeflow.acceleration.native.build`.
"""
import os
import typing as tp

import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops

_LIB_NAME = "_conditional_clade_op.so"
_module: tp.Optional[tp.Any] = None


def library_path() -> str:
    return os.path.join(os.path.dirname(__file__), _LIB_NAME)


def load_op_library():
    """Load (once) and return the compiled op library module."""
    global _module
    if _module is None:
        path = library_path()
        if not os.path.exists(path):
            raise RuntimeError(
                f"Native op library not found at {path}. "
                "Build it with treeflow/acceleration/native/build.sh "
                "(or `python -m treeflow.acceleration.native.build`)."
            )
        _module = tf.load_op_library(path)
        _register_gradient()
    return _module


def is_available() -> bool:
    """Return True if the native op library is built and loadable."""
    try:
        load_op_library()
        return True
    except Exception:
        return False


_gradient_registered = False


def _register_gradient():
    global _gradient_registered
    if _gradient_registered:
        return
    _gradient_registered = True

    @tf_ops.RegisterGradient("ConditionalCladeLogProb")
    def _conditional_clade_log_prob_grad(op, grad_log_prob, grad_flat_indices):
        # log_prob = sum of the chosen conditional log-probs, so the gradient
        # w.r.t. the conditional log-probs is a scatter-add of the upstream
        # gradient onto the chosen flat indices (the second output).
        conditional_log_probs = op.inputs[0]
        flat_indices = op.outputs[1]
        grad_cond = _module.conditional_clade_log_prob_grad(
            grad_log_prob, flat_indices, conditional_log_probs
        )
        # Inputs: conditional_log_probs, parent_indices, flat_parent, flat_child1.
        return [grad_cond, None, None, None]


def native_sample_parent_indices(
    logits: tf.Tensor,
    seeds: tf.Tensor,
    clade_offset: tf.Tensor,
    clade_count: tf.Tensor,
    flat_child1: tf.Tensor,
    flat_child2: tf.Tensor,
    taxon_count: int,
) -> tf.Tensor:
    """Sample ``parent_indices`` for one topology per row of ``seeds``.

    Parameters
    ----------
    logits
        ``[M]`` per-subsplit logits (need not be normalised; sampling is by a
        per-clade categorical over each segment).
    seeds
        ``[B, 2]`` int32 stateless-style seeds, one per sampled topology.
    clade_offset, clade_count
        ``[2**n]`` int32 per-clade segment offset and length, indexed by clade
        bitset.
    flat_child1, flat_child2
        ``[M]`` int32 child bitsets of each flat subsplit.

    Returns
    -------
    ``[B, 2n-2]`` int32 ``parent_indices``.
    """
    parent_indices, _flat = native_sample(
        logits, seeds, clade_offset, clade_count, flat_child1, flat_child2,
        taxon_count,
    )
    return parent_indices


def native_sample(
    logits: tf.Tensor,
    seeds: tf.Tensor,
    clade_offset: tf.Tensor,
    clade_count: tf.Tensor,
    flat_child1: tf.Tensor,
    flat_child2: tf.Tensor,
    taxon_count: int,
) -> tp.Tuple[tf.Tensor, tf.Tensor]:
    """Sample topologies, returning both ``parent_indices`` and ``flat_indices``.

    ``parent_indices`` is ``[B, 2n-2]`` (the TreeFlow topology encoding) and
    ``flat_indices`` is ``[B, n-1]`` -- the chosen flat subsplit index at each
    internal node, in expansion order. The order within a row is irrelevant
    downstream (the estimators sum over the ``n-1`` decisions), and these flat
    indices feed the traversal estimators directly, so a whole training step can
    run in graph mode.
    """
    module = load_op_library()
    return module.conditional_clade_sample(
        tf.convert_to_tensor(logits),
        tf.cast(seeds, tf.int32),
        tf.cast(clade_offset, tf.int32),
        tf.cast(clade_count, tf.int32),
        tf.cast(flat_child1, tf.int32),
        tf.cast(flat_child2, tf.int32),
        taxon_count=int(taxon_count),
    )


def native_topology_log_prob(
    conditional_log_probs: tf.Tensor,
    parent_indices: tf.Tensor,
    flat_parent: tf.Tensor,
    flat_child1: tf.Tensor,
    taxon_count: int,
) -> tf.Tensor:
    """Log-probability of each topology in ``parent_indices`` (``[B, 2n-2]``).

    Differentiable in ``conditional_log_probs`` (analytic scatter-add gradient).
    Returns a ``[B]`` tensor.
    """
    module = load_op_library()
    log_prob, _flat_indices = module.conditional_clade_log_prob(
        tf.convert_to_tensor(conditional_log_probs),
        tf.cast(parent_indices, tf.int32),
        tf.cast(flat_parent, tf.int32),
        tf.cast(flat_child1, tf.int32),
        taxon_count=int(taxon_count),
    )
    return log_prob


def native_parent_indices_to_child_indices(
    parent_indices: tf.Tensor, taxon_count: int
) -> tf.Tensor:
    """``child_indices`` (``[B, 2n-1, 2]``) from ``parent_indices`` (``[B, 2n-2]``)."""
    module = load_op_library()
    return module.parent_indices_to_child_indices(
        tf.cast(parent_indices, tf.int32), taxon_count=int(taxon_count)
    )


def native_child_indices_to_preorder(
    child_indices: tf.Tensor, taxon_count: int
) -> tf.Tensor:
    """Pre-order traversal (``[B, 2n-1]``) from ``child_indices`` (``[B, 2n-1, 2]``)."""
    module = load_op_library()
    return module.child_indices_to_preorder(
        tf.cast(child_indices, tf.int32), taxon_count=int(taxon_count)
    )


__all__ = [
    "native_sample",
    "native_sample_parent_indices",
    "native_topology_log_prob",
    "native_parent_indices_to_child_indices",
    "native_child_indices_to_preorder",
    "load_op_library",
    "is_available",
    "library_path",
]
