"""Timing harness shared by all benchmarkables.

Each computation is timed by calling it ``repeats`` times on a *single* fixed
input (one tree's actual branch lengths / ratios) and reporting both the
``mean`` and the ``min`` time across those repeats, rather than looping once
over a batch of different inputs. The traversal cost doesn't depend on the
branch-length values themselves, so repeating one input and averaging cancels
per-call dispatch jitter far better than a single timing per input does;
``min`` is the standard robust companion (discards positive scheduling noise,
never underestimates), so both are kept for comparison.

Ported from ``treeflow_benchmarks/benchmarking.py`` (generic, no BEAST
dependency).
"""
import typing as tp
from abc import abstractmethod
from collections import namedtuple
from timeit import default_timer as timer

import numpy as np


def time_function(func, *args, **kwargs):
    start = timer()
    res = func(*args, **kwargs)
    stop = timer()
    return stop - start, res


def repeated_times(func, *args, repeats: int = 20, **kwargs) -> tp.Tuple[np.ndarray, object]:
    """Call ``func(*args, **kwargs)`` ``repeats`` times, timing each call
    individually. Returns ``(times, last_result)``."""
    times = np.empty(repeats)
    result = None
    for i in range(repeats):
        times[i], result = time_function(func, *args, **kwargs)
    return times, result


def get_class_with_metadata(_class):
    return namedtuple(
        f"{_class.__name__}WithMetadata",
        _class._fields + ("taxon_count", "seed", "method", "model"),
    )


LikelihoodTimes = namedtuple("LikelihoodTimes", ["likelihood_time", "gradient_time"])
LikelihoodTimesWithMetadata = get_class_with_metadata(LikelihoodTimes)
RatioTransformTimes = namedtuple("RatioTransformTimes", ["forward_time", "gradient_time"])
RatioTransformTimesWithMetadata = get_class_with_metadata(RatioTransformTimes)

types_with_metadata = {
    LikelihoodTimes: LikelihoodTimesWithMetadata,
    RatioTransformTimes: RatioTransformTimesWithMetadata,
}


def annotate_times(times, taxon_count, seed, method, model):
    return types_with_metadata[type(times)](
        taxon_count=taxon_count, seed=seed, method=method, model=model, **times._asdict()
    )


class LikelihoodBenchmarkable:
    @abstractmethod
    def initialize(self, newick_file, fasta_file, model, calculate_clock_rate_gradient):
        pass

    @abstractmethod
    def calculate_likelihoods(self, branch_lengths: np.ndarray, params: object) -> np.ndarray:
        pass

    @abstractmethod
    def calculate_gradients(self, branch_lengths: np.ndarray, params: object):
        pass


def benchmark_likelihood(
    newick_file,
    fasta_file,
    model,
    branch_lengths: np.ndarray,
    benchmarkable: LikelihoodBenchmarkable,
    calculate_clock_rate_gradient: bool = False,
    repeats: int = 20,
) -> tp.Dict[str, LikelihoodTimes]:
    """Time ``benchmarkable`` on a single ``branch_lengths`` vector, repeated
    ``repeats`` times. Returns ``{"mean": LikelihoodTimes(...), "min": LikelihoodTimes(...)}``.
    """
    benchmarkable.initialize(newick_file, fasta_file, model, calculate_clock_rate_gradient)
    from benchmarks.params import get_numpy_gradient_params_dict
    from treeflow.model.phylo_model import PhyloModel

    params = get_numpy_gradient_params_dict(
        PhyloModel(model), calculate_clock_rate_gradient=calculate_clock_rate_gradient
    )
    likelihood_times, _ = repeated_times(
        benchmarkable.calculate_likelihoods, branch_lengths, params, repeats=repeats
    )
    gradient_times, _ = repeated_times(
        benchmarkable.calculate_gradients, branch_lengths, params, repeats=repeats
    )
    return dict(
        mean=LikelihoodTimes(float(likelihood_times.mean()), float(gradient_times.mean())),
        min=LikelihoodTimes(float(likelihood_times.min()), float(gradient_times.min())),
    )


class RatioTransformBenchmarkable:
    @abstractmethod
    def initialize(self, newick_file):
        pass

    @abstractmethod
    def calculate_heights(self, ratios):
        pass

    @abstractmethod
    def calculate_ratio_gradients(self, ratios, height_gradients):
        pass


def benchmark_ratio_transform(
    newick_file,
    ratios: np.ndarray,
    benchmarkable: RatioTransformBenchmarkable,
    repeats: int = 20,
) -> tp.Dict[str, RatioTransformTimes]:
    """Time ``benchmarkable`` on a single ``ratios`` vector, repeated ``repeats``
    times. Returns ``{"mean": RatioTransformTimes(...), "min": RatioTransformTimes(...)}``.
    """
    benchmarkable.initialize(newick_file)
    height_gradients = np.ones_like(ratios)

    forward_times, _ = repeated_times(benchmarkable.calculate_heights, ratios, repeats=repeats)
    gradient_times, _ = repeated_times(
        benchmarkable.calculate_ratio_gradients, ratios, height_gradients, repeats=repeats
    )
    return dict(
        mean=RatioTransformTimes(float(forward_times.mean()), float(gradient_times.mean())),
        min=RatioTransformTimes(float(forward_times.min()), float(gradient_times.min())),
    )
