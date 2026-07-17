"""Native (C++) subsplit Bayesian network topology sampler.

Compiled counterpart of the NumPy reference walk in
:meth:`treeflow.vbpi.sbn.SubsplitBayesianNetwork._sample_numpy`. Ancestral
sampling of a topology is a sequential traversal of the SBN's pointer-array
support (start at the root clade, draw a candidate child subsplit from the
pre-normalised conditional probabilities, recurse into non-leaf children), so it
is compiled into a single custom op rather than expressed as vectorised
TensorFlow -- mirroring the other native traversal ops here.

There is no gradient: topologies are discrete and the SBN's parameter gradients
flow through the differentiable ``log_prob`` (pure TensorFlow), so only the
forward draw is implemented.
"""
import os
import typing as tp

import numpy as np
import tensorflow as tf

_LIB_NAME = "_sbn_op.so"
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
    return _module


def is_available() -> bool:
    """Return True if the native op library is built and loadable."""
    try:
        load_op_library()
        return True
    except Exception:
        return False


def sample(
    child_offsets: np.ndarray,
    candidate_left_clade: np.ndarray,
    candidate_right_clade: np.ndarray,
    candidate_probs: np.ndarray,
    clade_leaf_taxon: np.ndarray,
    root_clade_id: int,
    taxon_count: int,
    n_samples: int,
    seed: int = 0,
) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample topologies from an SBN's pointer-array support.

    Parameters mirror :class:`treeflow.vbpi.support.SubsplitSupport`'s arrays
    plus the (already normalised, within each parent block) per-candidate
    conditional probabilities ``candidate_probs``.

    Returns
    -------
    parent_indices
        ``int32`` ``[n_samples, 2n-2]`` treeflow topologies.
    candidate_indices
        ``int32`` ``[n_samples, n-1]`` candidate index chosen per internal node.
    node_clade_ids
        ``int32`` ``[n_samples, 2n-1]`` support clade id per node.
    """
    module = load_op_library()
    child_offsets = tf.constant(np.asarray(child_offsets, dtype=np.int32))
    candidate_left_clade = tf.constant(
        np.asarray(candidate_left_clade, dtype=np.int32)
    )
    candidate_right_clade = tf.constant(
        np.asarray(candidate_right_clade, dtype=np.int32)
    )
    candidate_probs = tf.constant(np.asarray(candidate_probs, dtype=np.float64))
    clade_leaf_taxon = tf.constant(np.asarray(clade_leaf_taxon, dtype=np.int32))

    parent_indices, candidate_indices, node_clade_ids = module.sbn_sample(
        child_offsets,
        candidate_left_clade,
        candidate_right_clade,
        candidate_probs,
        clade_leaf_taxon,
        root_clade_id=int(root_clade_id),
        taxon_count=int(taxon_count),
        n_samples=int(n_samples),
        seed=int(seed),
    )
    return (
        parent_indices.numpy(),
        candidate_indices.numpy(),
        node_clade_ids.numpy(),
    )


__all__ = ["sample", "load_op_library", "is_available", "library_path"]
