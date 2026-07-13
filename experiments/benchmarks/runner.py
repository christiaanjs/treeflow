"""Run the simulate -> benchmark sweep and assemble a long-format DataFrame,
matching the ``out/plot-data.csv`` produced by the old Snakemake pipeline
(columns: ``method``, ``seed``, ``taxon_count``, ``model``, ``computation``,
``time``), plus a ``stat`` column distinguishing the ``mean`` and ``min``
across repeated timings (see ``benchmarking.repeated_times``).
"""
import os
import typing as tp

import numpy as np
import pandas as pd
import yaml

from benchmarks import benchmarking as bench
from benchmarks.benchmarkables import (
    build_likelihood_benchmarkables,
    build_ratio_transform_benchmarkables,
)
from benchmarks.simulate import get_ratios, simulate_replicate

_NAN_LIKELIHOOD_TIMES = dict(
    mean=bench.LikelihoodTimes(np.nan, np.nan), min=bench.LikelihoodTimes(np.nan, np.nan)
)
_NAN_RATIO_TRANSFORM_TIMES = dict(
    mean=bench.RatioTransformTimes(np.nan, np.nan),
    min=bench.RatioTransformTimes(np.nan, np.nan),
)


def _rows_from_times_by_stat(times_by_stat, taxon_count, seed, method, model):
    rows = []
    for stat, times in times_by_stat.items():
        row = bench.annotate_times(
            times, taxon_count=taxon_count, seed=seed, method=method, model=model
        )._asdict()
        row["stat"] = stat
        rows.append(row)
    return rows


def run_likelihood_sweep(
    taxon_counts: tp.Sequence[int],
    seeds: tp.Sequence[int],
    models: tp.Dict[str, dict],
    calculate_clock_rate_gradient: tp.Dict[str, bool],
    pop_size: float,
    sampling_window: float,
    sequence_length: int,
    repeats: int,
    sim_model: dict,
    working_dir: str,
) -> pd.DataFrame:
    """Simulate a tree + alignment per ``(taxon_count, seed)``, then time every
    available likelihood benchmarkable (treeflow, treeflow_native, jax if
    installed, beagle_bito if installed) on that tree's own branch lengths,
    repeated ``repeats`` times, for every model.
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
            branch_lengths_np = tensor_tree.branch_lengths.numpy()

            for model_name, model in models.items():
                for method, benchmarkable in benchmarkables.items():
                    try:
                        times_by_stat = bench.benchmark_likelihood(
                            newick_file,
                            fasta_file,
                            model,
                            branch_lengths_np,
                            benchmarkable,
                            calculate_clock_rate_gradient=calculate_clock_rate_gradient[
                                model_name
                            ],
                            repeats=repeats,
                        )
                    except Exception as ex:  # pragma: no cover - defensive, keep sweep going
                        print(f"  {method}/{model_name}/{taxon_count}taxa/seed{seed} failed: {ex}")
                        times_by_stat = _NAN_LIKELIHOOD_TIMES
                    rows.extend(
                        _rows_from_times_by_stat(
                            times_by_stat, taxon_count, seed, method, model_name
                        )
                    )
    return (
        pd.DataFrame(rows)
        .melt(
            id_vars=["method", "seed", "taxon_count", "model", "stat"],
            var_name="computation",
            value_name="time",
        )
    )


def run_ratio_transform_sweep(
    taxon_counts: tp.Sequence[int],
    seeds: tp.Sequence[int],
    pop_size: float,
    sampling_window: float,
    repeats: int,
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
            ratios_np = get_ratios(tensor_tree).numpy()

            for method, benchmarkable in benchmarkables.items():
                try:
                    times_by_stat = bench.benchmark_ratio_transform(
                        newick_file, ratios_np, benchmarkable, repeats=repeats
                    )
                except Exception as ex:  # pragma: no cover
                    print(f"  {method}/ratio_transform/{taxon_count}taxa/seed{seed} failed: {ex}")
                    times_by_stat = _NAN_RATIO_TRANSFORM_TIMES
                rows.extend(
                    _rows_from_times_by_stat(times_by_stat, taxon_count, seed, method, "none")
                )
    return (
        pd.DataFrame(rows)
        .melt(
            id_vars=["method", "seed", "taxon_count", "model", "stat"],
            var_name="computation",
            value_name="time",
        )
    )


def fit_log_log_lines(plot_data: pd.DataFrame) -> pd.DataFrame:
    """Log-log scaling exponent (``slope``) and ``intercept`` per
    method/computation/model/stat, matching ``treeflowbenchmarksr::fitLogLogLine``."""
    rows = []
    grouped = plot_data.dropna(subset=["time"]).groupby(
        ["method", "computation", "model", "stat"]
    )
    for (method, computation, model, stat), group in grouped:
        if len(group) < 2 or group["taxon_count"].nunique() < 2:
            continue
        slope, intercept = np.polyfit(
            np.log(group["taxon_count"]), np.log(group["time"]), deg=1
        )
        rows.append(
            dict(
                method=method, computation=computation, model=model, stat=stat,
                slope=slope, intercept=intercept,
            )
        )
    return pd.DataFrame(rows)


# --- Manuscript export -------------------------------------------------------
#
# The treeflow-paper manuscript build (``workflow/ms.smk`` -> the
# ``benchmark_plot`` R script and ``treeflow_pipeline.manuscript.
# benchmark_summary_table``) consumes a ``plot-data.csv`` / ``fit-table.csv``
# pair in the schema the *old* ``treeflow-benchmarks`` Snakemake pipeline
# produced:
#
#   plot-data.csv : method, seed, taxon_count, model, computation, time
#   fit-table.csv : method, computation, model, slope, intercept
#
# with ``computation`` in {``likelihood_time``, ``phylo_gradients_time``} and
# ``model`` in {``jc``, ``full``}. The richer frames produced above additionally
# carry ``stat`` (mean/min) and ``task`` columns, name the gradient computation
# ``gradient_time``, and include the ratio-transform task. ``write_manuscript_data``
# projects them back onto the manuscript schema so the figure and table can be
# regenerated directly from this benchmark.

_MANUSCRIPT_COMPUTATION_RENAME = {"gradient_time": "phylo_gradients_time"}
_MANUSCRIPT_COMPUTATIONS = ["likelihood_time", "phylo_gradients_time"]
_MANUSCRIPT_MODELS = ["jc", "full"]
# Methods shown in the manuscript figure/table. The direct bito benchmarkable
# (beagle_bito_direct) is used as the single "bito/BEAGLE" series -- matching the
# old pipeline and giving BEAGLE's compute without TensorFlow wrapper overhead;
# the tf.function-wrapped beagle_bito and jax_jit stay in the notebook's own
# exploratory plots (jax_jit excluded because the manuscript frames JAX as eager).
MANUSCRIPT_METHODS = ["treeflow", "treeflow_native", "beagle_bito_direct", "jax"]


def write_manuscript_data(
    plot_data: pd.DataFrame,
    fit_table: pd.DataFrame,
    out_dir: str,
    stat: str = "min",
    methods: tp.Optional[tp.Sequence[str]] = MANUSCRIPT_METHODS,
) -> tp.Tuple[str, str]:
    """Write ``manuscript-plot-data.csv`` and ``manuscript-fit-table.csv`` under
    ``out_dir`` in the schema consumed by the treeflow-paper manuscript build,
    restricted to the likelihood task and a single timing ``stat``. Only
    ``methods`` present in the data are kept (so bito/jax are included when they
    were available at run time and silently dropped otherwise). Returns the two
    written paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    ld = plot_data[(plot_data["task"] == "likelihood") & (plot_data["stat"] == stat)].copy()
    ld["computation"] = ld["computation"].replace(_MANUSCRIPT_COMPUTATION_RENAME)
    if methods is not None:
        ld = ld[ld["method"].isin(methods)]
    plot_path = os.path.join(out_dir, "manuscript-plot-data.csv")
    ld[["method", "seed", "taxon_count", "model", "computation", "time"]].to_csv(
        plot_path, index=False
    )

    ft = fit_table[fit_table["stat"] == stat].copy()
    ft["computation"] = ft["computation"].replace(_MANUSCRIPT_COMPUTATION_RENAME)
    # Restrict to the manuscript's likelihood-task computations and models; this
    # also drops the ratio-transform rows (model "none"), whose gradient_time is
    # renamed to phylo_gradients_time above and would otherwise slip through.
    ft = ft[
        ft["computation"].isin(_MANUSCRIPT_COMPUTATIONS)
        & ft["model"].isin(_MANUSCRIPT_MODELS)
    ]
    if methods is not None:
        ft = ft[ft["method"].isin(methods)]
    fit_path = os.path.join(out_dir, "manuscript-fit-table.csv")
    ft[["method", "computation", "model", "slope", "intercept"]].to_csv(
        fit_path, index=False
    )
    return plot_path, fit_path


def write_benchmark_config(
    taxon_counts: tp.Sequence[int],
    replicates: int,
    sequence_length: int,
    repeats: int,
    out_dir: str,
) -> str:
    """Write ``benchmark-config.yaml`` capturing the sweep parameters the
    manuscript quotes in its benchmark section (``treeflow_pipeline.manuscript.
    get_treeflow_manuscript_vars``). ``sample_count`` maps to ``repeats`` -- the
    number of times each computation is timed on a fixed input."""
    os.makedirs(out_dir, exist_ok=True)
    config = dict(
        full_taxon_counts=list(taxon_counts),
        replicates=replicates,
        sequence_length=sequence_length,
        sample_count=repeats,
    )
    path = os.path.join(out_dir, "benchmark-config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    return path
