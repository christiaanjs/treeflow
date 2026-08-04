"""Shared plumbing for fixed-topology MCMC samplers.

Both samplers in this package -- :mod:`treeflow.mcmc.hmc` and
:mod:`treeflow.mcmc.random_walk` -- work the same way: map the model's constrained
variables (including the tree, through the node-height ratio bijector) to a flat
list of unconstrained tensors, sample there, and map back. This module holds the
part that is identical between them.
"""

from functools import partial
import typing as tp

import tensorflow as tf
from tensorflow_probability.python.distributions import Distribution

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.model.event_shape_bijector import (
    get_fixed_topology_event_shape,
    get_fixed_topology_event_shape_and_space_bijector,
    get_unconstrained_init_values,
)
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology


class UnconstrainedTarget(tp.NamedTuple):
    """An unconstrained-space view of a pinned phylogenetic model."""

    #: ``(*unconstrained_parts) -> log density``, including the Jacobian term.
    target_log_prob_fn: tp.Callable[..., tf.Tensor]
    #: Initial unconstrained state, one tensor per variable.
    init_parts: tp.List[tf.Tensor]
    #: Variable names, in the order of ``init_parts``.
    names: tp.List[str]
    #: The constraining bijector, for mapping samples back.
    bijector: object

    def constrain(self, unconstrained_parts: tp.Sequence[tf.Tensor]):
        """Map a list of unconstrained sample tensors back to model variables."""
        return self.bijector.forward(dict(zip(self.names, unconstrained_parts)))


def get_unconstrained_target(
    model: Distribution,
    topologies: tp.Dict[str, TensorflowTreeTopology],
    init_state: tp.Optional[tp.Dict[str, object]] = None,
    num_chains: tp.Optional[int] = None,
    dtype=DEFAULT_FLOAT_DTYPE_TF,
) -> UnconstrainedTarget:
    """Build the unconstrained target density and an initial state for it.

    Parameters
    ----------
    model
        Pinned joint distribution representing the phylogenetic model.
    topologies
        Dict mapping tree variable names to fixed tree topologies.
    init_state
        Optional dict of constrained initial values (same format as ``init_loc``
        in the VI code). Unmapped variables start at zero in unconstrained space.
    num_chains
        When given, the initial state is tiled to a leading chain dimension of
        this size, so a sampler can run that many chains in parallel.
    """
    bijector, base_event_shape = get_fixed_topology_event_shape_and_space_bijector(
        model, topologies
    )
    names = list(base_event_shape.keys())

    def target_log_prob_fn(*unconstrained_parts):
        unconstrained_dict = dict(zip(names, unconstrained_parts))
        constrained = bijector.forward(unconstrained_dict)
        log_prob = model.unnormalized_log_prob(constrained)
        log_det_jacobian = bijector.forward_log_det_jacobian(
            unconstrained_dict,
            event_ndims={name: 1 for name in names},
        )
        return log_prob + log_det_jacobian

    event_shape_fn = partial(get_fixed_topology_event_shape, topology_pins=topologies)
    init_unconstrained = get_unconstrained_init_values(
        model, bijector, event_shape_fn=event_shape_fn, init=init_state
    )
    init_parts = [
        (
            tf.zeros(base_event_shape[name], dtype=dtype)
            if init_unconstrained[name] is None
            else tf.cast(init_unconstrained[name], dtype)
        )
        for name in names
    ]
    if num_chains is not None:
        init_parts = [
            tf.tile(tf.expand_dims(part, 0), [num_chains] + [1] * len(part.shape))
            for part in init_parts
        ]

    return UnconstrainedTarget(
        target_log_prob_fn=target_log_prob_fn,
        init_parts=init_parts,
        names=names,
        bijector=bijector,
    )


__all__ = ["get_unconstrained_target", "UnconstrainedTarget"]
