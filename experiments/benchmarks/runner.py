"""Run the simulate -> benchmark sweep and assemble a long-format DataFrame,
matching the ``out/plot-data.csv`` produced by the old Snakemake pipeline
(columns: ``method``, ``seed``, ``taxon_count``, ``model``, ``computation``,
``time``).
"""
import os
import typing as tp

import numpy as np
import pandas as pd

from benchmarks import benchmarking as bench
from benchmarks.benchmarkables import (
    build_likelihood_benchmarkables,
    build_ratio_transform_benchmarkables,
)
from benchmarks.simulate import simulate_height_samples, simulate_replicate


def run_likelihood_sweep(
    taxon_counts: tp.Sequence[int],
    seeds: tp.Sequence[int],
    models: tp.Dict[str, dict],
    calculate_clock_rate_gradient: tp.Dict[str, bool],
    pop_size: float,
    sampling_window: float,
    sequence_length: int,
    sample_count: int,
    height_scale: float,
    sim_model: dict,
    working_dir: str,
) -> pd.DataFrame:
    """Simulate a tree + alignment per ``(taxon_count, seed)``, then run every
    available likelihood benchmarkable (treeflow, treeflow_native, jax if
    installed, beagle_bito if installed) on it for every model.
    """
    from treeflow.model.phylo_model import PhyloModel

    rows = []
    for taxon_count in taxon_counts:
        benchmarkables = build_likelihood_benchmarkables()
        for seed in seeds:
            tree_dir = os.path.join(working_dir, f"{taxon_count}-taxa", f"{seed}-seed")
            newick_file, fasta_file, tensor_tree = simulate_replicate(
                taxon_count=taxon_count,
                pop_size=pop_size,
                sampling_window=sampling_window,
                sim_model=PhyloModel(sim_model),
                sequence_length=sequence_length,
                seed=seed,
                tree_dir=tree_dir,
            )
            branch_lengths, _ = simulate_height_samples(
                tensor_tree, sample_count=sample_count, height_scale=height_scale, seed=seed
            )
            branch_lengths_np = branch_lengths.numpy()

            for model_name, model in models.items():
                for method, benchmarkable in benchmarkables.items():
                    try:
                        times = bench.benchmark_likelihood(
                            newick_file,
                            fasta_file,
                            model,
                            branch_lengths_np,
                            benchmarkable,
                            calculate_clock_rate_gradient=calculate_clock_rate_gradient[
                                model_name
                            ],
                        )
                    except Exception as ex:  # pragma: no cover - defensive, keep sweep going
                        print(f"  {method}/{model_name}/{taxon_count}taxa/seed{seed} failed: {ex}")
                        times = bench.LikelihoodTimes(np.nan, np.nan)
                    rows.append(
                        bench.annotate_times(
                            times, taxon_count=taxon_count, seed=seed, method=method,
                            model=model_name,
                        )._asdict()
                    )
    return (
        pd.DataFrame(rows)
        .melt(
            id_vars=["method", "seed", "taxon_count", "model"],
            var_name="computation",
            value_name="time",
        )
    )


def run_ratio_transform_sweep(
    taxon_counts: tp.Sequence[int],
    seeds: tp.Sequence[int],
    pop_size: float,
    sampling_window: float,
    sample_count: int,
    height_scale: float,
    sim_model: dict,
    working_dir: str,
) -> pd.DataFrame:
    from treeflow.model.phylo_model import PhyloModel

    rows = []
    for taxon_count in taxon_counts:
        benchmarkables = build_ratio_transform_benchmarkables()
        for seed in seeds:
            tree_dir = os.path.join(working_dir, f"{taxon_count}-taxa", f"{seed}-seed")
            newick_file, _, tensor_tree = simulate_replicate(
                taxon_count=taxon_count,
                pop_size=pop_size,
                sampling_window=sampling_window,
                sim_model=PhyloModel(sim_model),
                sequence_length=10,  # sequences unused for this task
                seed=seed,
                tree_dir=tree_dir,
            )
            _, ratios = simulate_height_samples(
                tensor_tree, sample_count=sample_count, height_scale=height_scale, seed=seed
            )
            ratios_np = ratios.numpy()

            for method, benchmarkable in benchmarkables.items():
                try:
                    times = bench.benchmark_ratio_transform(newick_file, ratios_np, benchmarkable)
                except Exception as ex:  # pragma: no cover
                    print(f"  {method}/ratio_transform/{taxon_count}taxa/seed{seed} failed: {ex}")
                    times = bench.RatioTransformTimes(np.nan, np.nan)
                rows.append(
                    bench.annotate_times(
                        times, taxon_count=taxon_count, seed=seed, method=method, model="none"
                    )._asdict()
                )
    return (
        pd.DataFrame(rows)
        .melt(
            id_vars=["method", "seed", "taxon_count", "model"],
            var_name="computation",
            value_name="time",
        )
    )


def fit_log_log_lines(plot_data: pd.DataFrame) -> pd.DataFrame:
    """Log-log scaling exponent (``slope``) and ``intercept`` per
    method/computation/model, matching ``treeflowbenchmarksr::fitLogLogLine``."""
    rows = []
    grouped = plot_data.dropna(subset=["time"]).groupby(["method", "computation", "model"])
    for (method, computation, model), group in grouped:
        if len(group) < 2 or group["taxon_count"].nunique() < 2:
            continue
        slope, intercept = np.polyfit(
            np.log(group["taxon_count"]), np.log(group["time"]), deg=1
        )
        rows.append(
            dict(method=method, computation=computation, model=model, slope=slope,
                 intercept=intercept)
        )
    return pd.DataFrame(rows)
