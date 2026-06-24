"""Phylogenetic likelihood differentiable w.r.t. the child selection.

The Felsenstein pruning likelihood combines, at each internal node, the partial
likelihoods of its two *children*. Which nodes are the children is a discrete
property of the topology, normally applied as an integer **gather**
(``vals[children]`` in :mod:`treeflow.traversal.postorder`). A gather is exactly a
one-hot matrix multiplication ``onehot(child) @ all_partials`` -- so if we route a
custom gradient through it, the likelihood becomes differentiable w.r.t. the
*selection*, and a straight-through relaxation of the topology choice can push the
likelihood's gradient back into a clade-probability model.

We do **not** want to pay for a dense one-hot multiply, though: under a
straight-through estimator the forward pass is just the hard gather. So the
production primitive here, :func:`straight_through_gather`, runs a plain gather on
the forward pass and a ``tf.custom_gradient`` backward pass that sends gradients
back *as if* the selection had been applied by a one-hot ``@`` matmul -- with no
dense one-hot ever materialised.

:func:`relaxed_phylogenetic_likelihood` runs the postorder combine through either
that gather primitive (``gather=True``, the efficient path) or an explicit dense
one-hot ``@`` (``gather=False``). The dense path exists purely so the efficient
path's gradients can be validated against autodiff -- the two must agree.

This is a *separate* path from the production
:class:`~treeflow.distributions.leaf_ctmc.LeafCTMC` integer-gather likelihood,
which is untouched and stays the default for fixed-topology inference; the
selection-differentiable path is opt-in, for when gradients must reach the
topology / clade model.

Materialisation note (see the notebook discussion): the gradient w.r.t. a
selection row is ``<upstream, candidate_partial_k>`` over the candidate children
``k`` -- the *informative* part is the contrast between the realised child and the
**alternatives**, so the candidate child partials must exist. ``straight_through_gather``
avoids materialising the dense one-hot and the soft forward, but the candidate
partials it differentiates against are inherent; supplying them exactly needs the
visited clade's alternative subsplits (enumerable supports), and is otherwise
approximate (sampled / contrastive).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from treeflow.traversal.phylo_likelihood import _combine_child_partials


@tf.custom_gradient
def straight_through_gather(values: tf.Tensor, selection: tf.Tensor):
    """Gather rows of ``values`` selected by ``selection``, with a one-hot-multiply
    gradient.

    Parameters
    ----------
    values
        ``[num_candidates, *rest]`` candidate rows (e.g. the partial likelihoods or
        transition matrices of the candidate children).
    selection
        ``[out, num_candidates]`` selection weights. Its ``argmax`` along the last
        axis picks the gathered row (the forward pass is a hard gather, so
        ``selection`` should be one-hot in the forward direction -- e.g. a
        straight-through one-hot ``probs + stop_gradient(onehot - probs)``).

    Returns
    -------
    ``[out, *rest]`` gathered rows. The backward pass returns gradients *as if*
    ``out = selection @ values`` had been computed (so it matches autodiff of the
    dense one-hot multiply), without ever materialising that product on the
    forward pass.
    """
    values = tf.convert_to_tensor(values)
    selection = tf.convert_to_tensor(selection)
    index = tf.argmax(selection, axis=-1)
    out = tf.gather(values, index, axis=0)

    def grad(d_out):
        rest_axes = list(range(1, len(d_out.shape)))
        # d_values[k, *rest] = sum_o selection[o, k] d_out[o, *rest]  (= selection^T @ d_out)
        d_values = tf.tensordot(selection, d_out, axes=[[0], [0]])
        # d_selection[o, k] = <d_out[o], values[k]>  (= d_out @ values^T over *rest)
        d_selection = tf.tensordot(d_out, values, axes=[rest_axes, rest_axes])
        return d_values, d_selection

    return out, grad


def child_selection_from_topology(
    topology, dtype: tf.DType = tf.float64
) -> tf.Tensor:
    """One-hot child selection ``[n-1, 2, node_count]`` from a topology.

    Row ``u - n`` selects internal node ``u``'s two children among all nodes.
    Feeding this into :func:`relaxed_phylogenetic_likelihood` reproduces the
    standard integer-gather likelihood exactly (the relaxation "switched off").
    """
    child_indices = np.asarray(tf.get_static_value(topology.child_indices))
    taxon_count = int(tf.get_static_value(topology.taxon_count))
    node_count = 2 * taxon_count - 1
    selection = np.zeros((taxon_count - 1, 2, node_count))
    for node in range(taxon_count, node_count):
        c0, c1 = child_indices[node]
        selection[node - taxon_count, 0, int(c0)] = 1.0
        selection[node - taxon_count, 1, int(c1)] = 1.0
    return tf.constant(selection, dtype=dtype)


def relaxed_phylogenetic_likelihood(
    child_selection: tf.Tensor,
    sequences_onehot: tf.Tensor,
    transition_probs: tf.Tensor,
    frequencies: tf.Tensor,
    taxon_count: int,
    gather: bool = True,
) -> tf.Tensor:
    """Per-site phylogenetic likelihood differentiable w.r.t. ``child_selection``.

    Parameters
    ----------
    child_selection
        ``[n-1, 2, node_count]`` weights; ``child_selection[u-n]`` selects internal
        node ``u``'s two children among all nodes. One-hot reproduces the exact
        likelihood; the gradient w.r.t. these weights is what a straight-through
        topology relaxation rides on.
    sequences_onehot
        ``[..., leaf, state]`` one-hot leaf sequences.
    transition_probs
        ``[node, ..., state, state]`` per-edge transition matrices (node axis
        first), the matrix on the edge above each node.
    frequencies
        ``[..., state]`` root state frequencies.
    taxon_count
        Number of leaves ``n``.
    gather
        ``True`` (default, efficient): gather the children via
        :func:`straight_through_gather` (hard forward, one-hot-multiply gradient).
        ``False``: an explicit dense one-hot ``@`` matmul -- the reference whose
        autodiff gradient the efficient path is validated against.

    Returns
    -------
    Per-site likelihood ``[...]`` (sum ``log`` over sites for the log-likelihood).
    """
    n = taxon_count
    node_count = 2 * n - 1

    def select(selection, values):
        if gather:
            return straight_through_gather(values, selection)
        return tf.tensordot(selection, values, axes=[[1], [0]])

    # Leaf partials: vals[i] = sequences_onehot[..., i, :]  -> [..., state]
    vals = [sequences_onehot[..., i, :] for i in range(n)]
    for node in range(n, node_count):
        # Children of `node` have smaller indices (post-order labelling), so every
        # entry of `vals` needed below is already computed.
        selection = child_selection[node - n][:, :node]  # [2, node]
        child_partials = select(selection, tf.stack(vals, axis=0))  # [2, ..., state]
        child_transition_probs = select(
            selection, transition_probs[:node]
        )  # [2, ..., state, state]
        vals.append(
            _combine_child_partials(
                child_transition_probs, child_partials, use_matvec=False
            )
        )
    return tf.reduce_sum(frequencies * vals[node_count - 1], axis=-1)


__all__ = [
    "straight_through_gather",
    "child_selection_from_topology",
    "relaxed_phylogenetic_likelihood",
]
