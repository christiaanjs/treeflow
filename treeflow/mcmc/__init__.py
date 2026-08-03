"""MCMC samplers for fixed-topology Bayesian phylogenetics.

Both samplers map the model's constrained variables -- including the tree,
through the node-height ratio bijector -- to unconstrained space, sample there,
and map back (see :mod:`treeflow.mcmc.util`):

* :func:`fit_fixed_topology_hmc` -- Hamiltonian Monte Carlo or NUTS, using the
  target's gradients;
* :func:`fit_fixed_topology_random_walk_metropolis` -- random-walk
  Metropolis-Hastings, which uses no gradients and so makes a useful independent
  reference posterior for judging a variational approximation, together with the
  effective-sample-size diagnostics that say whether it can be trusted.
"""

from treeflow.mcmc.hmc import (
    fit_fixed_topology_hmc,
    HMCResults,
    KERNEL_HMC,
    KERNEL_NUTS,
)
from treeflow.mcmc.random_walk import (
    fit_fixed_topology_random_walk_metropolis,
    RandomWalkResults,
    check_effective_sample_size,
    effective_sample_size_summary,
)
from treeflow.mcmc.util import get_unconstrained_target, UnconstrainedTarget

__all__ = [
    "fit_fixed_topology_hmc",
    "HMCResults",
    "KERNEL_HMC",
    "KERNEL_NUTS",
    "fit_fixed_topology_random_walk_metropolis",
    "RandomWalkResults",
    "check_effective_sample_size",
    "effective_sample_size_summary",
    "get_unconstrained_target",
    "UnconstrainedTarget",
]
