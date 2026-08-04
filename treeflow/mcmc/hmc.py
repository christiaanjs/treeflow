from collections import namedtuple
import typing as tp
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import Distribution
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.mcmc.util import get_unconstrained_target

HMCResults = namedtuple("HMCResults", ("samples", "trace"))

KERNEL_HMC = "hmc"
KERNEL_NUTS = "nuts"


def fit_fixed_topology_hmc(
    model: Distribution,
    topologies: tp.Dict[str, TensorflowTreeTopology],
    num_results: int,
    num_burnin_steps: int,
    step_size: float = 0.01,
    num_leapfrog_steps: int = 10,
    num_adaptation_steps: tp.Optional[int] = None,
    init_state: tp.Optional[tp.Dict[str, object]] = None,
    seed: tp.Optional[int] = None,
    kernel: str = KERNEL_HMC,
) -> HMCResults:
    """Run Hamiltonian Monte Carlo for fixed-topology Bayesian phylogenetic inference.

    Parameters
    ----------
    model
        Pinned joint distribution representing the phylogenetic model.
    topologies
        Dict mapping tree variable names to fixed tree topologies.
    num_results
        Number of MCMC samples to collect after burn-in.
    num_burnin_steps
        Number of burn-in steps (discarded).
    step_size
        Initial leapfrog step size.
    num_leapfrog_steps
        Number of leapfrog steps per HMC proposal (ignored for NUTS).
    num_adaptation_steps
        Number of steps for dual-averaging step size adaptation.
        Defaults to ``num_burnin_steps``.
    init_state
        Optional dict of constrained initial values (same format as ``init_loc``
        in the VI code).  Unmapped variables start at zero in unconstrained space.
    seed
        Optional integer random seed.
    kernel
        Kernel type: ``"hmc"`` (default) or ``"nuts"``.

    Returns
    -------
    HMCResults
        Named tuple with ``samples`` (constrained samples as structured output
        from the model bijector, with batch shape ``[num_results]``) and
        ``trace`` (raw kernel results from ``sample_chain``).
    """
    if num_adaptation_steps is None:
        num_adaptation_steps = num_burnin_steps

    target = get_unconstrained_target(model, topologies, init_state=init_state)

    # Build MCMC kernel
    if kernel == KERNEL_NUTS:
        inner_kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=target.target_log_prob_fn,
            step_size=step_size,
        )
    else:
        inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target.target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
        )

    if num_adaptation_steps > 0:
        mcmc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=inner_kernel,
            num_adaptation_steps=num_adaptation_steps,
        )
    else:
        mcmc_kernel = inner_kernel

    samples_unconstrained, trace = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=target.init_parts,
        kernel=mcmc_kernel,
        seed=seed,
    )

    # Transform samples back to constrained space
    return HMCResults(samples=target.constrain(samples_unconstrained), trace=trace)


__all__ = ["fit_fixed_topology_hmc", "HMCResults", "KERNEL_HMC", "KERNEL_NUTS"]
