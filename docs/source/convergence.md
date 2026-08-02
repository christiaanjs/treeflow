# Plotting variational parameter traces

TreeFlow's variational inference (VI) works by *optimising* an approximation to the
posterior distribution: it repeatedly takes stochastic gradient steps that increase the
evidence lower bound (ELBO). Unlike MCMC, there is no notion of mixing or effective sample
size; "convergence" means the optimisation has run long enough that the ELBO has stopped
increasing and the parameters of the variational approximation have settled.

The main tool for checking that is the `treeflow_vi plot` subcommand, which draws the
optimisation trajectory of every variational parameter from a saved trace. It is a thin
wrapper around `plot_parameter_traces` from the
[`treeflow.vi.plotting`](treeflow.vi.plotting.rst) module, which can also be called
directly from Python.

## Saving a trace

`treeflow_vi plot` reads a trace saved by `treeflow_vi run --trace-output`:

```sh
treeflow_vi run \
    --input alignment.fasta \
    --topology tree.nwk \
    --model-file model.yaml \
    --num-steps 40000 \
    --trace-output trace.pkl
```

The trace is pickled as a `treeflow.vi.util.VIResults` named tuple:

* `loss` — a 1-D array with the loss (the negative ELBO contribution) at each step.
* `parameters` — a `dict` mapping variational parameter name to an array of shape
  `(num_steps, *param_shape)`. This is what `treeflow_vi plot` plots.
* `parameter_coords` — which coordinates of each parameter the trace holds; empty unless
  the run used `--max-trace-coords` (see below).

## `treeflow_vi plot`

```sh
treeflow_vi plot --trace trace.pkl --output trace.png
```

`--trace`/`-t` and `--output`/`-o` are the only required options; the output format is
inferred from the file extension (e.g. `.png`, `.pdf`) and `--dpi` (default 150) sets the
resolution. The command forces the non-interactive `Agg` matplotlib backend, so it works
on headless machines and over SSH without a display.

There are two layouts.

### Full layout (`--full`, the default)

One subplot per variational parameter, arranged in a grid of `--ncols` columns (default
3). Each subplot is titled with the parameter name and its flattened coordinate count,
e.g. `full_rank_scale_raw:0  (4096)`.

How a parameter is drawn depends on how many coordinates it has:

* **At most `--max-individual-lines` coordinates** (default 16): one line per coordinate.
* **More than that**: a shaded min–max envelope across all coordinates at each step, with
  a handful of evenly-spaced coordinate trajectories overlaid on top. This keeps large
  full-rank scale matrices and IAF weight matrices legible instead of drawing thousands of
  overlapping lines.

This is the layout to use for a single run: it shows every parameter, so nothing that is
still drifting can hide.

```sh
treeflow_vi plot -t trace.pkl -o full.png --ncols 4 --max-individual-lines 8
```

### Sampled layout (`--sample`)

A small, labelled, representative set of coordinates drawn into a *single* axis:

* **Scalar parameters** contribute their one coordinate.
* **Node-height vectors** — any parameter whose name contains a substring given by
  `--tree-vars` (default `tree`) — contribute the root plus evenly-spaced internal nodes.
  The root is the parameter's **last** coordinate; `--tree-coords` (default 3) sets the
  total, so the default draws the root plus two internal nodes.
* **Any other vector** contributes `--coords-per-var` (default 1) evenly-spaced
  coordinates.

Labels drop the TensorFlow `:0` suffix and annotate the coordinate by its index *in the
parameter*, e.g. `tree_loc[root]`, `tree_loc[node 12]`, `clock_rate_loc`. `--title` sets
the axis title.

```sh
treeflow_vi plot -t trace.pkl -o sample.png --sample --tree-coords 5 --title "run 1"
```

This layout is deliberately compact: it is meant for comparing several runs against each
other, which is easiest from Python (see below).

Options only apply to one layout: `--coords-per-var`, `--tree-vars`, `--tree-coords` and
`--title` are `--sample` only; `--max-individual-lines` and `--ncols` are `--full` only.

## Parameter names in the plots

The keys of `VIResults.parameters` are raw `tf.Variable` names, so what you see depends on
the `--variational-approximation` used for the run:

* `mean_field` — one location and scale variable per model variable, named after it:
  `tree_loc:0`, `tree_scale:0`, `clock_rate_loc:0`, ...
* `full_rank` — a single packed `full_rank_loc:0` vector over all free model dimensions
  and a `full_rank_scale_raw:0` matrix that is D × D in those dimensions.
* `root_full_rank` — a full-covariance block plus separate mean-field variables for
  anything passed to `--mean-field-vars`.
* `iaf` — the weights and biases of the flow's layers.

Because `full_rank` packs every model variable into one vector, its coordinate indices are
positions in that packed vector rather than named model variables, and `--tree-vars` will
not match it. To get per-variable blocks for a packed approximation, build your own
`name -> trace` mapping (slicing `full_rank_loc:0` into the blocks you care about) and call
`plot_parameter_traces` directly.

## Traces recorded with `--max-trace-coords`

`--trace-output` records the *entire* value of every parameter at every step, so the trace
is O(steps × parameter size). For large approximations — most notably `full_rank`, whose
scale matrix is quadratic in the number of free model dimensions — that can dominate memory
use on long runs over large trees. `treeflow_vi run --max-trace-coords N` instead records
up to `N` randomly-selected flattened coordinates per parameter (parameters with at most
`N` elements are still traced in full), making the trace O(steps × N).

Such traces plot exactly the same way, and remain correctly labelled: the trace records
which coordinates it kept (in `VIResults.parameter_coords`), so `treeflow_vi plot` reports
each line's index in the parameter rather than its position in the trace. The last
coordinate of a parameter — the root, for a node-height vector — is always among those
kept, so `tree_loc[root]` in the `--sample` layout is the root height whatever else was
sampled. Full-layout subplot titles show both counts, e.g.
`full_rank_scale_raw:0  (50 of 4096)`.

The remaining loss is resolution: everything between the sampled coordinates is simply not
recorded, so raise `--max-trace-coords` if a plot looks too coarse to judge.

## Using `plot_parameter_traces` from Python

```python
import pickle
import matplotlib.pyplot as plt
from treeflow.vi.plotting import plot_parameter_traces

with open("trace.pkl", "rb") as f:
    trace = pickle.load(f)

axes = plot_parameter_traces(trace.parameters)   # full layout; array of Axes
axes[0].figure.savefig("full.png", dpi=150, bbox_inches="tight")
```

The function is approximation-agnostic: each entry of the mapping is a trace with a leading
step axis and an arbitrary trailing shape, and trailing dimensions are flattened to
coordinates. `sample=False` (the default) returns the flat array of `Axes` for the full
layout; `sample=True` returns the single `Axes` it drew into.

Passing `ax` (sampled layout) or `axes` (full layout) draws into existing axes instead of
creating a figure, which is how several runs are compared side by side:

```python
fig, axs = plt.subplots(1, 3, figsize=(15, 3.5), sharex=True)
for ax, path in zip(axs, ["run1.pkl", "run2.pkl", "run3.pkl"]):
    with open(path, "rb") as f:
        run = pickle.load(f)
    plot_parameter_traces(run.parameters, sample=True, ax=ax, title=path)
fig.tight_layout()
```

Other keyword arguments mirror the CLI options: `coords_per_var`, `tree_vars`,
`tree_coords`, `max_individual_lines`, `ncols`, and `figsize_per_plot` for the size of each
subplot in a created full-layout figure. An empty `parameter_trace` raises `ValueError`.
The full signature is in the [`treeflow.vi.plotting`](treeflow.vi.plotting.rst) API
reference, and the trace containers in
[`treeflow.vi.util`](treeflow.vi.util.rst) (`VIResults`, `TracedCoordinates`).

For a trace written with `--max-trace-coords`, pass `parameter_coords=trace.parameter_coords`
alongside `trace.parameters` — that is what lets coordinates be labelled by their index in
the parameter (and the root be found) rather than by their position in the trace. It is
harmless to pass for a full trace, where it is empty.

## The ELBO trace

`treeflow_vi plot` covers the parameters only; the ELBO itself lives in `VIResults.loss`
and is one line of matplotlib:

```python
plt.plot(-trace.loss)          # ELBO over optimisation
plt.xlabel("Iteration")
plt.ylabel("ELBO")
```

`treeflow_vi run` also prints an ELBO estimate computed from the last `--elbo-samples`
steps (default 100). It is the quantity being maximised, so a higher (less negative) value
is better, and checking that it stops improving as `--num-steps` grows is the cheapest
convergence check.

## Reading the plots

A converged run looks like this:

* the ELBO trace has flattened into a stable plateau (aside from stochastic noise) rather
  than still trending upward, and
* every variational parameter trace has stopped drifting — flat lines, and for summarised
  parameters a min–max envelope of constant width.

If either is still changing at the end of the run, increase `--num-steps`; if the traces
are very noisy or unstable, also decrease `--learning-rate`. Rather than restarting from
scratch, continue the previous run with `--resume-from-trace`, which warm-starts the
variational parameters from the last step of a saved trace (the model, topology and
`--variational-approximation` must match; optimizer state such as Adam moments is not
restored):

```sh
treeflow_vi run \
    --input alignment.fasta \
    --topology tree.nwk \
    --model-file model.yaml \
    --num-steps 40000 \
    --resume-from-trace trace.pkl \
    --trace-output trace-continued.pkl
```

Note that `--convergence-criterion nonfinite`, the criterion exposed by the CLI, stops a
run when the loss, gradients or parameters become non-finite. It is a **divergence guard**,
not an ELBO-plateau detector, so convergence still has to be confirmed by inspecting the
traces as above.

Finally, remember that VI approximates the posterior: converged traces tell you the
optimisation has finished, not that the approximating family captures the posterior
exactly. Compare marginals against a reference MCMC analysis where feasible, especially for
divergence times and other strongly correlated parameters.
