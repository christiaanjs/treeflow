"""Timing harness shared by all benchmarkables.

Ported from ``treeflow_benchmarks/benchmarking.py`` (generic, no BEAST
dependency).
"""
import typing as tp
from abc import abstractmethod
from collections import namedtuple
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf


def time_function(func, *args, **kwargs):
    start = timer()
    res = func(*args, **kwargs)
    stop = timer()
    return stop - start, res


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

    def calculate_likelihoods_loop(self, branch_lengths: np.ndarray, params: object):
        batch_size = branch_lengths.shape[0]
        output = np.zeros(batch_size, dtype=branch_lengths.dtype)
        for i in range(batch_size):
            output[i] = self.calculate_likelihoods(branch_lengths[i], params)
        return output

    def calculate_gradients_loop(self, branch_lengths: np.ndarray, params: object):
        batch_size = branch_lengths.shape[0]
        for i in range(batch_size):
            self.calculate_gradients(branch_lengths[i], params)


def benchmark_likelihood(
    newick_file,
    fasta_file,
    model,
    branch_lengths: np.ndarray,
    benchmarkable: LikelihoodBenchmarkable,
    calculate_clock_rate_gradient: bool = False,
) -> LikelihoodTimes:
    benchmarkable.initialize(newick_file, fasta_file, model, calculate_clock_rate_gradient)
    from benchmarks.params import get_numpy_gradient_params_dict
    from treeflow.model.phylo_model import PhyloModel

    params = get_numpy_gradient_params_dict(
        PhyloModel(model), calculate_clock_rate_gradient=calculate_clock_rate_gradient
    )
    likelihood_time, _ = time_function(
        benchmarkable.calculate_likelihoods_loop, branch_lengths, params
    )
    gradient_time, _ = time_function(
        benchmarkable.calculate_gradients_loop, branch_lengths, params
    )
    return LikelihoodTimes(likelihood_time, gradient_time)


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
    newick_file, ratios: np.ndarray, benchmarkable: RatioTransformBenchmarkable
) -> RatioTransformTimes:
    benchmarkable.initialize(newick_file)
    height_gradients = np.ones_like(ratios)

    forward_time, _ = time_function(benchmarkable.calculate_heights, ratios)
    gradient_time, _ = time_function(
        benchmarkable.calculate_ratio_gradients, ratios, height_gradients
    )
    return RatioTransformTimes(forward_time, gradient_time)
