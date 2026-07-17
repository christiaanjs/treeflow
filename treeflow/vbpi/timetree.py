"""Time-tree building blocks for VBPI / topology MCMC.

The VBPI comparison is done over **time trees**: rooted, (here) ultrametric trees
parameterised by internal-node *heights* (times), with a coalescent or
birth-death prior over those times. This module wires the VBPI stack to
TreeFlow's existing time-tree machinery:

* node heights are unconstrained through TreeFlow's
  :class:`~treeflow.bijectors.node_height_ratio_bijector.NodeHeightRatioChainBijector`
  (the ``NodeHeightRatioTransform``): a real vector ``z`` maps to valid node
  heights for *any* topology, which is exactly what lets the same latent be
  reused as topologies change;
* the tree prior is an existing TreeFlow distribution --
  :class:`~treeflow.distributions.tree.coalescent.constant_coalescent.ConstantCoalescent`
  or :class:`~treeflow.distributions.tree.birthdeath.yule.Yule`;
* the likelihood reuses the native Jukes-Cantor pruning
  (:func:`treeflow.vbpi.likelihood.rooted_jc_log_likelihood`) with branch lengths
  ``clock_rate * (parent_height - child_height)`` read off the rooted tree.

:class:`NodeHeightRatioModel` is the variational ``q(heights | T)`` used by VBPI,
amortised across topologies by keying each node's height parameter on its clade
id (the same pointer-array clade ids the SBN uses).
"""
import typing as tp

import numpy as np
import tensorflow as tf

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.bijectors.node_height_ratio_bijector import (
    NodeHeightRatioChainBijector,
)
from treeflow.distributions.tree.birthdeath.yule import Yule
from treeflow.distributions.tree.coalescent.constant_coalescent import (
    ConstantCoalescent,
)
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
from treeflow.tree.topology.tensorflow_tree_topology import (
    TensorflowTreeTopology,
    numpy_topology_to_tensor,
)
from treeflow.vbpi.likelihood import rooted_jc_log_likelihood
from treeflow.vbpi.support import SubsplitSupport


def as_topology(topology) -> TensorflowTreeTopology:
    """Coerce ``parent_indices`` (or a topology) to a ``TensorflowTreeTopology``."""
    if isinstance(topology, TensorflowTreeTopology):
        return topology
    return numpy_topology_to_tensor(
        NumpyTreeTopology(parent_indices=np.asarray(topology))
    )


def build_time_tree(
    topology: TensorflowTreeTopology,
    node_heights: tf.Tensor,
    sampling_times: tp.Optional[tf.Tensor] = None,
) -> TensorflowRootedTree:
    """Assemble a :class:`TensorflowRootedTree` from a topology and node heights.

    ``sampling_times`` defaults to zeros (homochronous / ultrametric leaves).
    """
    node_heights = tf.convert_to_tensor(node_heights, dtype=DEFAULT_FLOAT_DTYPE_TF)
    taxon_count = int(node_heights.shape[-1]) + 1
    if sampling_times is None:
        sampling_times = tf.zeros(taxon_count, dtype=node_heights.dtype)
    return TensorflowRootedTree(
        node_heights=node_heights,
        sampling_times=sampling_times,
        topology=topology,
    )


def time_tree_jc_log_likelihood(
    topology: TensorflowTreeTopology,
    leaf_partials: tf.Tensor,
    node_heights: tf.Tensor,
    clock_rate: float = 1.0,
    sampling_times: tp.Optional[tf.Tensor] = None,
    use_native: tp.Union[bool, str] = "auto",
) -> tf.Tensor:
    """JC log likelihood of a time tree (branch lengths from node heights).

    Branch lengths in expected substitutions are
    ``clock_rate * (parent_height - child_height)``, read off the rooted tree, and
    scored with the native pruning op.
    """
    tree = build_time_tree(topology, node_heights, sampling_times)
    branch_lengths = tf.convert_to_tensor(clock_rate, dtype=tree.branch_lengths.dtype)
    branch_lengths = branch_lengths * tree.branch_lengths
    return rooted_jc_log_likelihood(
        None, leaf_partials, branch_lengths, topology=topology, use_native=use_native
    )


def coalescent_prior(
    taxon_count: int, pop_size: tf.Tensor, sampling_times: tp.Optional[tf.Tensor] = None
) -> ConstantCoalescent:
    if sampling_times is None:
        sampling_times = tf.zeros(taxon_count, dtype=DEFAULT_FLOAT_DTYPE_TF)
    return ConstantCoalescent(taxon_count, pop_size, sampling_times)


def yule_prior(taxon_count: int, birth_rate: tf.Tensor) -> Yule:
    return Yule(taxon_count, birth_rate)


class NodeHeightRatioModel:
    """Amortised variational ``q(node heights | T)`` in unconstrained ratio space.

    A diagonal Gaussian is placed on the unconstrained node-height-ratio vector
    ``z`` (``NodeHeightRatioChainBijector`` maps ``z`` to valid heights for the
    tree). Its per-node ``loc`` / ``scale`` are keyed by the clade id of each
    internal node, so -- exactly like :class:`SplitLognormalBranchModel` -- the
    parameters are shared across topologies. Sampling is reparameterised, and
    ``log q`` is returned in *height* space (Gaussian density minus the transform
    log-det-Jacobian), ready to enter the VBPI importance weights alongside the
    coalescent/Yule prior.
    """

    def __init__(
        self,
        support: SubsplitSupport,
        loc: tp.Optional[tf.Variable] = None,
        scale_param: tp.Optional[tf.Variable] = None,
        init_loc: float = 0.0,
        init_scale: float = 0.5,
        dtype: tf.DType = DEFAULT_FLOAT_DTYPE_TF,
        use_native: tp.Union[bool, str] = False,
    ):
        self.support = support
        self.dtype = dtype
        # The forward transform is differentiated (through heights and, via the
        # ELBO, through its own log-det-Jacobian); the native ratio op registers
        # only a first-order gradient, so default to the pure-TensorFlow path.
        self.use_native = use_native
        num = support.num_clades
        if loc is None:
            loc = tf.Variable(
                tf.fill([num], tf.constant(init_loc, dtype=dtype)), name="height_loc"
            )
        if scale_param is None:
            inv = np.log(np.expm1(init_scale))  # softplus^{-1}(init_scale)
            scale_param = tf.Variable(
                tf.fill([num], tf.constant(inv, dtype=dtype)),
                name="height_scale_param",
            )
        self.loc = loc
        self.scale_param = scale_param

    def scale(self) -> tf.Tensor:
        return tf.nn.softplus(tf.convert_to_tensor(self.scale_param, self.dtype))

    def _internal_clade_ids(self, node_clade_ids: tf.Tensor) -> tf.Tensor:
        """Clade id of each internal node, ordered by internal id (root last)."""
        taxon_count = self.support.taxon_count
        return node_clade_ids[..., taxon_count:]  # [..., n-1]

    def sample_and_log_prob(
        self,
        node_clade_ids: tf.Tensor,
        topologies: tp.Sequence[TensorflowTreeTopology],
        seed=None,
    ) -> tp.Tuple[tf.Tensor, tf.Tensor]:
        """Reparameterised node heights and their ``log q`` (height space).

        Parameters
        ----------
        node_clade_ids
            ``[K, 2n-1]`` per-node clade ids for the ``K`` sampled trees.
        topologies
            The ``K`` corresponding ``TensorflowTreeTopology`` objects (the ratio
            transform is topology specific).

        Returns
        -------
        node_heights
            ``[K, n-1]`` internal node heights.
        log_prob
            ``[K]`` ``log q(node heights)``.
        """
        internal_clade_ids = self._internal_clade_ids(node_clade_ids)  # [K, n-1]
        loc = tf.gather(tf.convert_to_tensor(self.loc, self.dtype), internal_clade_ids)
        scale = tf.gather(self.scale(), internal_clade_ids)
        eps = tf.random.stateless_normal(
            tf.shape(loc), seed=_as_seed(seed), dtype=self.dtype
        )
        z = loc + scale * eps  # [K, n-1]
        # Diagonal-Gaussian log density of z, summed over nodes.
        log_q_z = tf.reduce_sum(
            -0.5 * ((z - loc) / scale) ** 2
            - tf.math.log(scale)
            - 0.5 * tf.math.log(tf.constant(2.0 * np.pi, dtype=self.dtype)),
            axis=-1,
        )
        heights = []
        fldj = []
        for k, topology in enumerate(topologies):
            bijector = NodeHeightRatioChainBijector(
                topology, use_native=self.use_native
            )
            heights.append(bijector.forward(z[k]))
            fldj.append(bijector.forward_log_det_jacobian(z[k], event_ndims=1))
        node_heights = tf.stack(heights, axis=0)  # [K, n-1]
        fldj = tf.stack(fldj, axis=0)  # [K]
        log_prob = log_q_z - fldj  # density in height space
        return node_heights, log_prob


def _as_seed(seed):
    if seed is None:
        return tf.random.uniform([2], maxval=2**30, dtype=tf.int32)
    if isinstance(seed, int):
        return tf.constant([seed, 0], dtype=tf.int32)
    return tf.convert_to_tensor(seed, dtype=tf.int32)


__all__ = [
    "as_topology",
    "build_time_tree",
    "time_tree_jc_log_likelihood",
    "coalescent_prior",
    "yule_prior",
    "NodeHeightRatioModel",
]
