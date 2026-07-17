"""Per-split branch-length variational approximation for VBPI.

VBPI factorises the branch-length posterior over the *splits* (edges) of a tree
and places an independent diagonal log-normal on each. Because two different
topologies that share an edge share that edge's parameters, the branch-length
approximation is *amortised* across topologies: the parameters are keyed by the
clade sitting below the edge, i.e. by the same clade ids that index the SBN's
pointer-array structure (:class:`SubsplitSupport`).

Concretely, for a sampled topology with node-clade ids ``c_i`` (the support
clade id of node ``i``), the branch above node ``i`` is::

    b_i ~ LogNormal( loc[c_i], softplus(scale_param[c_i]) )

drawn with the reparameterisation trick so gradients flow to ``loc`` and
``scale_param``. The root has no branch, so only the ``2n-2`` non-root nodes
contribute. Sampling and ``log_prob`` return quantities summed over a tree's
branches, ready to enter the VBPI importance weights.
"""
import typing as tp

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.vbpi.support import SubsplitSupport

tfd = tfp.distributions


class SplitLognormalBranchModel:
    """Amortised diagonal-log-normal branch lengths keyed by clade id."""

    def __init__(
        self,
        support: SubsplitSupport,
        loc: tp.Optional[tf.Variable] = None,
        scale_param: tp.Optional[tf.Variable] = None,
        init_loc: float = -2.0,
        init_scale: float = 0.1,
        dtype: tf.DType = DEFAULT_FLOAT_DTYPE_TF,
    ):
        """
        Parameters
        ----------
        support
            The SBN support; supplies ``num_clades`` (one parameter per clade)
            and the clade-id layout the samples are expressed in.
        loc, scale_param
            Optional pre-existing variables of shape ``[num_clades]``. ``loc`` is
            the log-normal location; the scale is ``softplus(scale_param)`` so it
            stays positive without a constraint.
        init_loc
            Initial ``loc`` (default ``-2`` ~ median branch length ``0.14``).
        init_scale
            Initial branch-length scale; ``scale_param`` is initialised to its
            softplus inverse so ``softplus(scale_param) == init_scale``.
        """
        self.support = support
        self.dtype = dtype
        num = support.num_clades
        if loc is None:
            loc = tf.Variable(
                tf.fill([num], tf.constant(init_loc, dtype=dtype)),
                name="branch_loc",
            )
        if scale_param is None:
            inv = np.log(np.expm1(init_scale))  # softplus^{-1}(init_scale)
            scale_param = tf.Variable(
                tf.fill([num], tf.constant(inv, dtype=dtype)),
                name="branch_scale_param",
            )
        self.loc = loc
        self.scale_param = scale_param

    def scale(self) -> tf.Tensor:
        return tf.nn.softplus(tf.convert_to_tensor(self.scale_param, self.dtype))

    def _branch_clade_ids(self, node_clade_ids: tf.Tensor) -> tf.Tensor:
        """Clade id of each of the ``2n-2`` non-root branches.

        ``node_clade_ids`` is ``[..., 2n-1]`` indexed by node id; the root (the
        last id) has no branch, so we drop the final column.
        """
        node_count = 2 * self.support.taxon_count - 1
        return node_clade_ids[..., : node_count - 1]

    def _distribution(self, node_clade_ids: tf.Tensor) -> tfd.Distribution:
        branch_clade_ids = self._branch_clade_ids(node_clade_ids)
        loc = tf.gather(tf.convert_to_tensor(self.loc, self.dtype), branch_clade_ids)
        scale = tf.gather(self.scale(), branch_clade_ids)
        return tfd.Independent(
            tfd.LogNormal(loc=loc, scale=scale),
            reinterpreted_batch_ndims=1,
        )

    def sample_and_log_prob(
        self,
        node_clade_ids: tf.Tensor,
        seed=None,
    ) -> tp.Tuple[tf.Tensor, tf.Tensor]:
        """Reparameterised branch lengths and their joint ``log q``.

        Parameters
        ----------
        node_clade_ids
            Integer tensor ``[..., 2n-1]`` of per-node clade ids (from the
            sampler's ``node_clade_ids``).

        Returns
        -------
        branch_lengths
            ``[..., 2n-2]`` reparameterised branch lengths, indexed by node id.
        log_prob
            ``[...]`` sum of per-branch log densities.
        """
        node_clade_ids = tf.convert_to_tensor(node_clade_ids)
        dist = self._distribution(node_clade_ids)
        branch_lengths = dist.sample(seed=seed)
        return branch_lengths, dist.log_prob(branch_lengths)

    def log_prob(
        self, node_clade_ids: tf.Tensor, branch_lengths: tf.Tensor
    ) -> tf.Tensor:
        """``log q`` of given branch lengths under the per-split approximation."""
        node_clade_ids = tf.convert_to_tensor(node_clade_ids)
        dist = self._distribution(node_clade_ids)
        return dist.log_prob(branch_lengths)

    def __repr__(self) -> str:
        return f"SplitLognormalBranchModel(support={self.support!r})"


__all__ = ["SplitLognormalBranchModel"]
