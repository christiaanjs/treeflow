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

## Layout

| module | contents |
| --- | --- |
| `simulate.py` | coalescent tree + sequence simulation (no BEAST), FASTA/newick writers |
| `params.py` | parameter-dict plumbing shared by the benchmarkables |
| `benchmarking.py` | generic timing harness |
| `benchmarkables.py` | treeflow / treeflow-native / jax / beagle-bito implementations |
| `runner.py` | simulate -> benchmark sweep -> long-format `pandas.DataFrame` |
| `data/` | `plot-data.csv`, `fit-table.csv` and the summary plots produced by the notebook |

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
