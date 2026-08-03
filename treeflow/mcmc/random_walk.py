"""Random-walk Metropolis-Hastings for fixed-topology phylogenetic inference.

The counterpart of :mod:`treeflow.mcmc.hmc` with a random-walk proposal instead of
a gradient-based one. It exists mainly as a *reference posterior* for judging
variational approximations: it makes no use of the target's gradients, so it
shares nothing with the machinery being evaluated, and its correctness does not
depend on any of the flow's transformations being right.

Sampling happens in the same unconstrained space the variational approximations
work in (see :mod:`treeflow.mcmc.util`), so a Gaussian proposal is a
reasonable one and the samples come back as ordinary constrained model
variables.

Because a random walk in tens of dimensions mixes slowly, **the effective sample
size is not optional here** -- it is the thing that says whether the reference
can be trusted. :func:`fit_fixed_topology_random_walk_metropolis` computes it for
every variable (and, with more than one chain, the potential scale reduction
``R-hat``), and :func:`check_effective_sample_size` turns that into a pass/fail
against a threshold.

The proposal scale is tuned during burn-in by a Robbins-Monro update on
``log(scale)`` towards a target acceptance rate (0.234, the optimal rate for a
random walk in high dimension), since TFP's step-size adaptation kernels do not
apply to ``RandomWalkMetropolis``. Burn-in also estimates a **per-coordinate**
proposal scale from the chain's own spread so far, which matters a great deal
when the coordinates differ in scale by orders of magnitude, as node heights do.
Preconditioning changes only the proposal, never the target, so the stationary
distribution is unaffected and the reference stays exact.
"""

from collections import namedtuple
import typing as tp
import warnings

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import Distribution

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.mcmc.util import get_unconstrained_target

RandomWalkResults = namedtuple(
    "RandomWalkResults",
    (
        "samples",
        "unconstrained_samples",
        "names",
        "acceptance_rate",
        "scale",
        "effective_sample_size",
        "potential_scale_reduction",
        "num_results",
        "num_chains",
    ),
)

#: Optimal acceptance rate for a random-walk proposal in high dimension.
DEFAULT_TARGET_ACCEPTANCE_RATE = 0.234
DEFAULT_ADAPTATION_CHUNK = 100
DEFAULT_ADAPTATION_GAIN = 5.0


def _make_chunk_sampler(target_log_prob_fn, base_scales, dtype, thin=0):
    """A compiled ``(scale, state, seed) -> (samples, is_accepted)`` chain runner.

    The proposal scale is a *tensor* argument rather than baked into the kernel,
    so the whole chain traces once and is reused across every adaptation chunk;
    running the chain eagerly instead costs a Python-level model evaluation per
    step, which is far too slow for a reference posterior.
    """

    @tf.function(reduce_retracing=True)
    def run(scale, current_state, num_steps, seed):
        kernel = tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=target_log_prob_fn,
            new_state_fn=tfp.mcmc.random_walk_normal_fn(
                scale=[tf.cast(scale, dtype) * base for base in base_scales]
            ),
        )
        return tfp.mcmc.sample_chain(
            num_results=num_steps,
            num_burnin_steps=0,
            num_steps_between_results=thin,
            current_state=current_state,
            kernel=kernel,
            trace_fn=lambda _, results: results.is_accepted,
            seed=seed,
        )

    return run


def _normalised_spread(states, dtype, floor=1e-3):
    """Per-coordinate standard deviation of a set of chain states, mean 1.

    Flattens the leading (step, chain) axes, floors the result so a coordinate
    that has barely moved cannot freeze, and normalises so that the separately
    tuned global scale keeps its meaning.
    """
    flat = tf.reshape(states, tf.concat([[-1], tf.shape(states)[-1:]], axis=0))
    spread = tf.math.reduce_std(flat, axis=0)
    spread = tf.maximum(spread, floor * tf.reduce_max(spread))
    return tf.cast(spread / tf.reduce_mean(spread), dtype)


def fit_fixed_topology_random_walk_metropolis(
    model: Distribution,
    topologies: tp.Dict[str, TensorflowTreeTopology],
    num_results: int,
    num_burnin_steps: int,
    scale: float = 0.1,
    num_chains: int = 1,
    thin: int = 1,
    base_scales: tp.Optional[tp.Sequence[float]] = None,
    adapt_scale: bool = True,
    precondition: bool = True,
    adaptation_chunk: int = DEFAULT_ADAPTATION_CHUNK,
    adaptation_gain: float = DEFAULT_ADAPTATION_GAIN,
    target_acceptance_rate: float = DEFAULT_TARGET_ACCEPTANCE_RATE,
    init_state: tp.Optional[tp.Dict[str, object]] = None,
    seed: tp.Optional[object] = None,
    dtype=DEFAULT_FLOAT_DTYPE_TF,
    progress_bar: tp.Optional[tp.Callable] = None,
) -> RandomWalkResults:
    """Sample the posterior by random-walk Metropolis-Hastings.

    Parameters
    ----------
    model
        Pinned joint distribution representing the phylogenetic model.
    topologies
        Dict mapping tree variable names to fixed tree topologies.
    num_results
        Number of samples to keep, per chain, after burn-in.
    num_burnin_steps
        Burn-in steps (discarded). When ``adapt_scale`` is set these are also the
        steps over which the proposal scale is tuned.
    scale
        Initial proposal standard deviation in unconstrained space.
    num_chains
        Chains to run in parallel. More than one enables the ``R-hat`` potential
        scale reduction diagnostic, which catches chains stuck in different
        places -- something the effective sample size of a single chain cannot.
    thin
        Keep one sample every ``thin`` steps. The chain still runs
        ``num_results * thin`` steps; thinning bounds the memory a long run
        needs without discarding much information, since successive random-walk
        states are highly correlated anyway.
    base_scales
        Optional multipliers on ``scale``, one per variable (in the order of the
        returned ``names``). Each may be a scalar or a per-coordinate vector.
        Replaced by the preconditioner unless ``precondition`` is off.
    adapt_scale
        Tune ``scale`` during burn-in towards ``target_acceptance_rate``.
    precondition
        Estimate per-coordinate proposal scales partway through burn-in (each
        coordinate's standard deviation over the burn-in states, normalised to
        mean 1). Only the proposal changes, so the chain still targets exactly
        the same posterior.
    adaptation_chunk, adaptation_gain
        Burn-in is run in chunks of ``adaptation_chunk`` steps; after each,
        ``log(scale)`` moves by ``adaptation_gain * (acceptance - target)``,
        damped as ``1/sqrt(chunk)``.
    init_state
        Optional dict of constrained initial values (same format as ``init_loc``
        in the VI code).
    seed
        Seed for the sampler.
    progress_bar
        Optional ``tqdm``-like callable, wrapped around the burn-in chunks and
        the sampling chunk.

    Returns
    -------
    RandomWalkResults
        ``samples`` are constrained model variables with batch shape
        ``[num_results]`` (or ``[num_results, num_chains]``);
        ``effective_sample_size`` and ``potential_scale_reduction`` are dicts
        keyed by variable name.
    """
    target = get_unconstrained_target(
        model, topologies, init_state=init_state, num_chains=num_chains, dtype=dtype
    )
    if base_scales is None:
        base_scales = [tf.ones([], dtype=dtype)] * len(target.names)
    else:
        base_scales = [tf.cast(base, dtype) for base in base_scales]

    seeds = tfp.random.split_seed(
        tfp.random.sanitize_seed(seed), n=max(1, num_burnin_steps // max(adaptation_chunk, 1)) + 2
    )
    seed_iter = iter(seeds)

    run_chunk = _make_chunk_sampler(target.target_log_prob_fn, base_scales, dtype)
    current_state = target.init_parts
    current_scale = tf.constant(scale, dtype=dtype)
    acceptance_rate = None

    # ---- Burn-in, tuning the proposal scale as we go ----
    if num_burnin_steps > 0:
        chunk_size = min(adaptation_chunk, num_burnin_steps) if adapt_scale else num_burnin_steps
        num_chunks = max(1, num_burnin_steps // chunk_size)
        chunks = range(num_chunks)
        if progress_bar is not None:
            chunks = progress_bar(chunks, desc="burn-in")
        precondition_after = num_chunks // 2 if precondition else None
        burnin_states = []
        adaptation_step = 0
        for chunk in chunks:
            samples, is_accepted = run_chunk(
                current_scale, current_state, chunk_size, next(seed_iter)
            )
            current_state = [part[-1] for part in samples]
            if precondition_after is not None:
                burnin_states.append(samples)
                if chunk + 1 >= precondition_after:
                    # Per-coordinate spread so far, normalised so the global
                    # scale keeps its meaning; the tuning below re-adapts it.
                    base_scales = [
                        _normalised_spread(
                            tf.concat([state[i] for state in burnin_states], axis=0),
                            dtype,
                        )
                        for i in range(len(target.names))
                    ]
                    run_chunk = _make_chunk_sampler(
                        target.target_log_prob_fn, base_scales, dtype
                    )
                    burnin_states = []
                    precondition_after = None
                    # The per-coordinate scales change what a given global scale
                    # means, so let it re-adapt at full gain rather than at the
                    # decayed rate it had reached.
                    adaptation_step = 0
            acceptance_rate = tf.reduce_mean(
                tf.cast(is_accepted, dtype)
            )
            if adapt_scale:
                # Robbins-Monro on log(scale): raise it when we accept too often
                # (proposals too timid), lower it when we accept too rarely.
                adaptation_step += 1
                gain = adaptation_gain / tf.sqrt(tf.cast(adaptation_step, dtype))
                current_scale = current_scale * tf.exp(
                    gain * (acceptance_rate - tf.cast(target_acceptance_rate, dtype))
                )

    # ---- Sampling ----
    if thin > 1:
        run_chunk = _make_chunk_sampler(
            target.target_log_prob_fn, base_scales, dtype, thin=thin - 1
        )
    samples, is_accepted = run_chunk(
        current_scale, current_state, num_results, next(seed_iter)
    )
    acceptance_rate = tf.reduce_mean(tf.cast(is_accepted, dtype))

    unconstrained = dict(zip(target.names, samples))
    cross_chain_dims = 1 if num_chains > 1 else None
    ess = {
        name: tfp.mcmc.effective_sample_size(
            part, filter_beyond_positive_pairs=True, cross_chain_dims=cross_chain_dims
        )
        for name, part in unconstrained.items()
    }
    if num_chains > 1:
        r_hat = {
            name: tfp.mcmc.potential_scale_reduction(part, independent_chain_ndims=1)
            for name, part in unconstrained.items()
        }
    else:
        r_hat = None

    return RandomWalkResults(
        samples=target.constrain(samples),
        unconstrained_samples=unconstrained,
        names=target.names,
        acceptance_rate=acceptance_rate,
        scale=current_scale,
        effective_sample_size=ess,
        potential_scale_reduction=r_hat,
        num_results=num_results,
        num_chains=num_chains,
    )


def effective_sample_size_summary(
    results: RandomWalkResults,
) -> tp.Dict[str, tp.Dict[str, float]]:
    """Per-variable worst-case ESS (and ``R-hat``), as plain floats."""
    summary = {}
    for name in results.names:
        ess = results.effective_sample_size[name].numpy()
        entry = {
            "min ESS": float(ess.min()),
            "median ESS": float(np.median(ess)) if ess.size else float("nan"),
            "ESS per sample": float(ess.min())
            / (results.num_results * results.num_chains),
        }
        if results.potential_scale_reduction is not None:
            entry["max R-hat"] = float(
                results.potential_scale_reduction[name].numpy().max()
            )
        summary[name] = entry
    return summary


def check_effective_sample_size(
    results: RandomWalkResults,
    min_ess: float = 200.0,
    max_r_hat: float = 1.05,
    raise_on_failure: bool = False,
) -> tp.Tuple[bool, tp.Dict[str, tp.Dict[str, float]]]:
    """Check the chain is mixed well enough to serve as a reference.

    Returns ``(passed, summary)``, warns on failure, and raises instead when
    ``raise_on_failure`` is set. A random walk over tens of node heights mixes
    slowly, so a failure here usually means "run it longer", not "the sampler is
    broken" -- but it does mean the run should not be quoted as ground truth.
    """
    summary = effective_sample_size_summary(results)
    failures = []
    for name, entry in summary.items():
        if entry["min ESS"] < min_ess:
            failures.append(f"{name}: min ESS {entry['min ESS']:.1f} < {min_ess}")
        if "max R-hat" in entry and entry["max R-hat"] > max_r_hat:
            failures.append(
                f"{name}: max R-hat {entry['max R-hat']:.3f} > {max_r_hat}"
            )
    passed = not failures
    if not passed:
        message = (
            "Random-walk chain has not mixed well enough to be a reference "
            f"posterior (acceptance rate {float(results.acceptance_rate):.3f}): "
            + "; ".join(failures)
        )
        if raise_on_failure:
            raise RuntimeError(message)
        warnings.warn(message)
    return passed, summary


__all__ = [
    "fit_fixed_topology_random_walk_metropolis",
    "RandomWalkResults",
    "effective_sample_size_summary",
    "check_effective_sample_size",
    "DEFAULT_TARGET_ACCEPTANCE_RATE",
]
