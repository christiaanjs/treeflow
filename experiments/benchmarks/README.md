# Benchmarks

Inlined, BEAST-free replacement for the
[`treeflow-benchmarks`](https://github.com/christiaanjs/treeflow-benchmarks)
Snakemake pipeline that produced the benchmark figure (Fig 7) in the TreeFlow
paper.

Run `../benchmark_pipeline.ipynb` (a Jupyter notebook, not a Snakemake
pipeline) to simulate trees + alignments and benchmark likelihood/gradient and
node-height ratio-transform computations across taxon counts and methods:

* `treeflow` -- plain TensorFlow-graph implementation
* `treeflow_native` -- the same code with `use_native=True`, routing through
  treeflow's compiled C++ ops (`treeflow.acceleration.native`)
* `jax`/`jax_jit` -- via [phylojax](https://github.com/christiaanjs/phylojax),
  if installed (both block on results so JAX's asynchronous dispatch does not
  hide compute time in the timings)
* `beagle_bito` -- via [bito](https://github.com/phylovi/bito)/BEAGLE, driven
  through TreeFlow's `tf.function` wrapper (so timings include the same
  TensorFlow dispatch/marshalling overhead as the `treeflow` methods), if
  installed
* `beagle_bito_direct` -- the same BEAGLE/bito instance driven *directly*
  (branch lengths written into bito's own state array, `inst.log_likelihoods()`
  / `inst.phylo_gradients()` called straight through), bypassing the TensorFlow
  wrapper to isolate BEAGLE's raw compute; if installed. This is the series the
  manuscript uses for bito/BEAGLE.

Install the extra dependencies with `pip install -e ".[benchmark]"` (adds
`pandas`, `matplotlib` and `jupyter`; `phylojax` and `bito` are independent
optional installs, detected at runtime).

## Running

Interactively, just run the notebook — it uses the `quick` profile by default
(minutes). To run the manuscript-scale sweep from the browser, set
`BENCHMARK_PROFILE=full` in the environment before launching the kernel.

To execute the notebook non-interactively with the sweep progress streamed live
to your terminal (`jupyter nbconvert --execute` swallows it), use the helper
script:

```bash
python ../run_benchmark.py --profile quick   # minutes; resume from checkpoints
python ../run_benchmark.py --profile full    # manuscript scale (hours; needs bito+jax)
python ../run_benchmark.py --force           # recompute everything
python ../run_benchmark.py --output run.ipynb --timeout 7200
```

The two profiles are:

| profile | taxon counts | replicates | sites | repeats |
| --- | --- | --- | --- | --- |
| `quick` | 8–128 | 3 | 200 | 10 |
| `full` (manuscript) | 32–2048 | 10 | 1000 | 100 (eager JAX capped at 512 taxa) |

### Checkpointing / resumption

Each `(taxon_count, seed, model, method)` config is cached to a per-config CSV
under `data/checkpoints/<task>/<params-hash>/` as it completes. Re-running the
notebook (or `run_benchmark.py`) **skips already-complete configs** — and skips
the tree simulation for a `(taxon_count, seed)` entirely when all of its configs
are cached — so an interrupted or extended sweep resumes where it left off,
Snakemake-style. The `<params-hash>` sub-directory keys the cache on the sweep
parameters that affect timings (sequence length, repeats, models, ...), so
changing them starts a fresh cache rather than reusing stale results. Pass
`force=True` (notebook `FORCE_RERUN`, or `run_benchmark.py --force`) to recompute
and overwrite. The checkpoint directory is git-ignored.

Slow methods can be capped to a maximum taxon count via `METHOD_MAX_TAXON_COUNT`
(e.g. `{"jax": 512}`, mirroring the old pipeline's `short_benchmarkables`); a
capped method is skipped above its limit and its line simply stops early.

## Layout

| module | contents |
| --- | --- |
| `simulate.py` | coalescent tree + sequence simulation (no BEAST), FASTA/newick writers |
| `params.py` | parameter-dict plumbing shared by the benchmarkables |
| `benchmarking.py` | generic timing harness |
| `benchmarkables.py` | treeflow / treeflow-native / jax / beagle-bito implementations |
| `runner.py` | simulate -> benchmark sweep (with per-config checkpointing) -> long-format `pandas.DataFrame` |
| `../run_benchmark.py` | execute the notebook via nbclient, streaming sweep progress to the terminal |
| `data/` | `plot-data.csv`, `fit-table.csv` and the summary plots produced by the notebook |
| `data/checkpoints/` | resumable per-config timing cache (git-ignored) |

## Notes on the simulation

`treeflow.distributions.tree.coalescent.constant_coalescent.ConstantCoalescent._sample_n`
is currently a placeholder ("Dummy sampling") rather than a real sampler, so
`simulate.simulate_coalescent_tree` implements the standard forwards-in-time
coalescent waiting-time algorithm directly (exponential coalescence times at
rate `k * (k - 1) / (2 * pop_size)`, interleaved with any remaining serial
sampling times). Sequence simulation reuses
`treeflow.model.phylo_model.get_sequence_distribution` -- the same function
used to build production likelihoods -- and samples from it, so simulated
data is guaranteed consistent with the models the benchmarks evaluate.
