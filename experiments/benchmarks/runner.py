"""Run the simulate -> benchmark sweep and assemble a long-format DataFrame,
matching the ``out/plot-data.csv`` produced by the old Snakemake pipeline
(columns: ``method``, ``seed``, ``taxon_count``, ``model``, ``computation``,
``time``), plus a ``stat`` column distinguishing the ``mean`` and ``min``
across repeated timings (see ``benchmarking.repeated_times``).
"""
import hashlib
import json
import os
import typing as tp

import numpy as np
import pandas as pd
import yaml

# Progress bar flavour: interactive sessions get the rich tqdm.auto widget bar,
# but under nbconvert (or when BENCHMARK_TQDM=text) we want a plain text bar that
# streams as stderr 'stream' messages -- which run_benchmark.py forwards live to
# the terminal. Widget bars would not forward as text.
if os.environ.get("BENCHMARK_TQDM", "").lower() in ("text", "std", "plain"):
    from tqdm.std import tqdm
else:
    from tqdm.auto import tqdm

from benchmarks import benchmarking as bench
from benchmarks.benchmarkables import (
    build_likelihood_benchmarkables,
    build_ratio_transform_benchmarkables,
)
from benchmarks.simulate import get_ratios, simulate_replicate

# Long-format columns each config contributes (shared by both sweeps and by the
# per-config checkpoint files).
_ID_VARS = ["method", "seed", "taxon_count", "model", "stat"]
_LONG_COLUMNS = _ID_VARS + ["computation", "time"]


def _signature(params: dict) -> str:
    """Short hash of the sweep parameters that affect timings. Used as a
    checkpoint sub-directory so a run with different parameters (sequence
    length, repeats, models, ...) doesn't reuse stale cached configs."""
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _config_checkpoint_path(checkpoint_dir, task, signature, taxon_count, seed, model, method):
    return os.path.join(
        checkpoint_dir,
        task,
        signature,
        f"{taxon_count}taxa-{seed}seed-{model}-{method}.csv",
    )


def _write_csv_atomic(frame: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)  # atomic: a half-written file never looks complete


def _melt_config(wide_rows) -> pd.DataFrame:
    return pd.DataFrame(wide_rows).melt(
        id_vars=_ID_VARS, var_name="computation", value_name="time"
    )

_NAN_LIKELIHOOD_TIMES = dict(
    mean=bench.LikelihoodTimes(np.nan, np.nan), min=bench.LikelihoodTimes(np.nan, np.nan)
)
_NAN_RATIO_TRANSFORM_TIMES = dict(
    mean=bench.RatioTransformTimes(np.nan, np.nan),
    min=bench.RatioTransformTimes(np.nan, np.nan),
)


def _method_included(
    method: str,
    taxon_count: int,
    method_max_taxon_count: tp.Optional[tp.Mapping[str, int]],
) -> bool:
    """Whether ``method`` should run at ``taxon_count``. A method listed in
    ``method_max_taxon_count`` is skipped above its cap (mirroring the old
    pipeline's ``short_benchmarkables``/``short_taxon_counts``, e.g. running the
    slow eager-JAX benchmark only up to 512 taxa); methods absent from the
    mapping always run."""
    if method_max_taxon_count is None:
        return True
    cap = method_max_taxon_count.get(method)
    return cap is None or taxon_count <= cap


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
    progress: bool = True,
    method_max_taxon_count: tp.Optional[tp.Mapping[str, int]] = None,
    checkpoint_dir: tp.Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    """Simulate a tree + alignment per ``(taxon_count, seed)``, then time every
    available likelihood benchmarkable (treeflow, treeflow_native, jax if
    installed, beagle_bito if installed) on that tree's own branch lengths,
    repeated ``repeats`` times, for every model.

    ``method_max_taxon_count`` optionally caps individual methods to a maximum
    taxon count (e.g. ``{"jax": 512}`` to skip the slow eager-JAX benchmark on
    large trees, as the old pipeline's ``short_benchmarkables`` did); a skipped
    config produces no rows, so that method's line simply stops early in the
    plots.

    ``checkpoint_dir`` enables Snakemake-style resumption: each
    ``(taxon_count, seed, model, method)`` config is written to its own CSV
    (under a sub-directory keyed by a hash of the sweep parameters) as it
    completes, and on a later run an existing config file is loaded instead of
    recomputed -- and the (potentially expensive) tree simulation for a
    ``(taxon_count, seed)`` is skipped entirely when all its configs are already
    cached. Pass ``force=True`` to recompute and overwrite regardless.

    A ``tqdm`` progress bar (``progress=True``) tracks every config and shows the
    live per-config min likelihood time; set ``progress=False`` to silence it.
    """
    from treeflow.model.phylo_model import PhyloModel

    method_names = list(build_likelihood_benchmarkables().keys())
    signature = _signature(
        dict(
            task="likelihood",
            sequence_length=sequence_length,
            repeats=repeats,
            pop_size=pop_size,
            sampling_window=sampling_window,
            sim_model=sim_model,
            models=models,
            calculate_clock_rate_gradient=calculate_clock_rate_gradient,
        )
    )

    def config_path(taxon_count, seed, model_name, method):
        if checkpoint_dir is None:
            return None
        return _config_checkpoint_path(
            checkpoint_dir, "likelihood", signature, taxon_count, seed, model_name, method
        )

    def is_cached(taxon_count, seed, model_name, method):
        path = config_path(taxon_count, seed, model_name, method)
        return (not force) and path is not None and os.path.exists(path)

    total = len(seeds) * len(models) * sum(
        _method_included(m, tc, method_max_taxon_count)
        for tc in taxon_counts
        for m in method_names
    )
    bar = tqdm(total=total, disable=not progress, desc="likelihood sweep", unit="cfg")

    frames = []
    for taxon_count in taxon_counts:
        benchmarkables = build_likelihood_benchmarkables()
        for seed in seeds:
            included = [
                (model_name, method)
                for model_name in models
                for method in benchmarkables
                if _method_included(method, taxon_count, method_max_taxon_count)
            ]
            # Simulate only if at least one config for this (taxon_count, seed)
            # still needs computing -- otherwise every result is loaded from disk.
            need_compute = any(
                not is_cached(taxon_count, seed, mn, me) for mn, me in included
            )
            if need_compute:
                bar.set_postfix_str(f"{taxon_count} taxa, seed {seed}: simulating")
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

            for model_name, method in included:
                path = config_path(taxon_count, seed, model_name, method)
                if is_cached(taxon_count, seed, model_name, method):
                    bar.set_postfix_str(
                        f"{taxon_count}taxa seed{seed} {model_name}/{method}: cached"
                    )
                    frames.append(pd.read_csv(path))
                    bar.update(1)
                    continue

                bar.set_postfix_str(f"{taxon_count}taxa seed{seed} {model_name}/{method}")
                try:
                    times_by_stat = bench.benchmark_likelihood(
                        newick_file,
                        fasta_file,
                        models[model_name],
                        branch_lengths_np,
                        benchmarkables[method],
                        calculate_clock_rate_gradient=calculate_clock_rate_gradient[
                            model_name
                        ],
                        repeats=repeats,
                    )
                    like = times_by_stat["min"].likelihood_time
                    grad = times_by_stat["min"].gradient_time
                    bar.set_postfix_str(
                        f"{taxon_count}taxa seed{seed} {model_name}/{method}: "
                        f"like {like * 1e3:.2f}ms grad {grad * 1e3:.2f}ms (min)"
                    )
                except Exception as ex:  # pragma: no cover - defensive, keep sweep going
                    tqdm.write(
                        f"  {method}/{model_name}/{taxon_count}taxa/seed{seed} "
                        f"failed: {ex}"
                    )
                    times_by_stat = _NAN_LIKELIHOOD_TIMES
                frame = _melt_config(
                    _rows_from_times_by_stat(
                        times_by_stat, taxon_count, seed, method, model_name
                    )
                )
                if path is not None:
                    _write_csv_atomic(frame, path)
                frames.append(frame)
                bar.update(1)
    bar.close()
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=_LONG_COLUMNS)


def run_ratio_transform_sweep(
    taxon_counts: tp.Sequence[int],
    seeds: tp.Sequence[int],
    pop_size: float,
    sampling_window: float,
    repeats: int,
    sim_model: dict,
    working_dir: str,
    progress: bool = True,
    method_max_taxon_count: tp.Optional[tp.Mapping[str, int]] = None,
    checkpoint_dir: tp.Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    """Time the node-height ratio transform forward pass and its gradient for
    every available benchmarkable. Supports the same ``method_max_taxon_count``
    caps and ``checkpoint_dir``/``force`` resumption as ``run_likelihood_sweep``.
    """
    from treeflow.model.phylo_model import PhyloModel

    method_names = list(build_ratio_transform_benchmarkables().keys())
    signature = _signature(
        dict(
            task="ratio_transform",
            repeats=repeats,
            pop_size=pop_size,
            sampling_window=sampling_window,
            sim_model=sim_model,
        )
    )

    def config_path(taxon_count, seed, method):
        if checkpoint_dir is None:
            return None
        return _config_checkpoint_path(
            checkpoint_dir, "ratio_transform", signature, taxon_count, seed, "none", method
        )

    def is_cached(taxon_count, seed, method):
        path = config_path(taxon_count, seed, method)
        return (not force) and path is not None and os.path.exists(path)

    total = len(seeds) * sum(
        _method_included(m, tc, method_max_taxon_count)
        for tc in taxon_counts
        for m in method_names
    )
    bar = tqdm(total=total, disable=not progress, desc="ratio-transform sweep", unit="cfg")

    frames = []
    for taxon_count in taxon_counts:
        benchmarkables = build_ratio_transform_benchmarkables()
        for seed in seeds:
            included = [
                method
                for method in benchmarkables
                if _method_included(method, taxon_count, method_max_taxon_count)
            ]
            need_compute = any(not is_cached(taxon_count, seed, me) for me in included)
            if need_compute:
                bar.set_postfix_str(f"{taxon_count} taxa, seed {seed}: simulating")
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

            for method in included:
                path = config_path(taxon_count, seed, method)
                if is_cached(taxon_count, seed, method):
                    bar.set_postfix_str(f"{taxon_count}taxa seed{seed} {method}: cached")
                    frames.append(pd.read_csv(path))
                    bar.update(1)
                    continue

                bar.set_postfix_str(f"{taxon_count}taxa seed{seed} {method}")
                try:
                    times_by_stat = bench.benchmark_ratio_transform(
                        newick_file, ratios_np, benchmarkables[method], repeats=repeats
                    )
                    fwd = times_by_stat["min"].forward_time
                    grad = times_by_stat["min"].gradient_time
                    bar.set_postfix_str(
                        f"{taxon_count}taxa seed{seed} {method}: "
                        f"fwd {fwd * 1e3:.2f}ms grad {grad * 1e3:.2f}ms (min)"
                    )
                except Exception as ex:  # pragma: no cover
                    tqdm.write(
                        f"  {method}/ratio_transform/{taxon_count}taxa/seed{seed} "
                        f"failed: {ex}"
                    )
                    times_by_stat = _NAN_RATIO_TRANSFORM_TIMES
                frame = _melt_config(
                    _rows_from_times_by_stat(times_by_stat, taxon_count, seed, method, "none")
                )
                if path is not None:
                    _write_csv_atomic(frame, path)
                frames.append(frame)
                bar.update(1)
    bar.close()
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=_LONG_COLUMNS)


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
