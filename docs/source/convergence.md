# Monitoring convergence of variational inference

TreeFlow's variational inference (VI) works by *optimising* an approximation to the
posterior distribution: it repeatedly takes stochastic gradient steps that increase the
evidence lower bound (ELBO). Unlike MCMC, there is no notion of mixing or effective sample
size; instead, "convergence" means that the optimisation has run for long enough that the
ELBO has stopped increasing and the parameters of the variational approximation have
settled. This page describes how to check that in TreeFlow.

## The ELBO estimate printed by the CLI

When you run `treeflow_vi`, the command prints, after optimisation finishes:

```
Ran inference for <N> iterations
ELBO estimate: <value>
```

The ELBO estimate is computed from the last `--elbo-samples` steps of the optimisation
(default 100). It is the quantity being maximised, so a higher (less negative) value is
better. Running the analysis with a larger `--num-steps` and checking that the ELBO
estimate stops improving is the simplest convergence check.

## Saving and plotting the optimisation trace

For a proper diagnosis, save the full optimisation trace with `--trace-output`:

```sh
treeflow_vi run \
    --input alignment.fasta \
    --topology tree.nwk \
    --model-file model.yaml \
    --num-steps 40000 \
    --trace-output trace.pkl
```

The trace is pickled as a `treeflow.vi.util.VIResults` named tuple with two fields:

* `loss` — a 1-D array with the loss (the negative ELBO contribution) at each optimisation
  step. The ELBO is the negative of this, so plotting `-loss` against the step number
  shows the ELBO over the course of optimisation.
* `parameters` — the values of the variational parameters at each step, which can be
  plotted to check that they have stopped drifting.

The `treeflow_vi plot` subcommand plots the parameter traces directly from a saved trace
file, using the `treeflow.vi.plotting.plot_parameter_traces` helper:

```sh
treeflow_vi plot --trace trace.pkl --output trace.png
```

By default this draws one subplot per variational parameter (`--full`, the default). Pass
`--sample` to instead draw a small representative set of coordinates into a single axis,
which is useful for comparing several runs side by side. See `treeflow_vi plot --help` for
the full set of options (e.g. `--coords-per-var`, `--tree-vars`, `--tree-coords`).

For the ELBO trace itself, a minimal plot:

```python
import pickle
import matplotlib.pyplot as plt

with open("trace.pkl", "rb") as f:
    trace = pickle.load(f)

plt.plot(-trace.loss)          # ELBO over optimisation
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.show()
```

You have a converged run when:

* the ELBO trace has flattened into a stable plateau (aside from stochastic noise) rather
  than still trending upward, and
* the traces of the individual variational parameters have stopped drifting.

If either is still changing at the end of the run, increase `--num-steps` (and, if the
trace is very noisy or unstable, decrease the learning rate with `--learning-rate`) and
run again.

## Resuming a run from a saved trace

If a run turns out not to have converged, or was interrupted, it can be continued rather
than restarted from scratch with `--resume-from-trace`, pointing at a trace saved by an
earlier `--trace-output`:

```sh
treeflow_vi run \
    --input alignment.fasta \
    --topology tree.nwk \
    --model-file model.yaml \
    --num-steps 40000 \
    --resume-from-trace trace.pkl \
    --trace-output trace-continued.pkl
```

This warm-starts the variational parameters from the last step of the given trace, so
optimisation continues from where the previous run left off instead of from a fresh
initialisation. The resumed run must use the same `--variational-approximation`, model and
topology as the run that produced the trace; the optimizer state itself (e.g. Adam moment
estimates) is not restored, only the variational parameters.

## Convergence criteria and the `--convergence-criterion` option

The `--convergence-criterion` option lets optimisation stop early. The currently available
criterion, `nonfinite`, stops the run if the loss, gradients or parameters become
non-finite; it is a **divergence guard**, not an ELBO-plateau detector. Automatic
detection of an ELBO plateau is not yet implemented, so convergence should be confirmed by
inspecting the ELBO and parameter traces as described above.

## Guidelines

* Start with a generous `--num-steps` and reduce it once you have seen where the ELBO
  plateaus for your data and model.
* Re-run with a different random seed (`--seed`) to confirm that the fitted approximation
  is reproducible.
* Remember that VI approximates the posterior: a converged run tells you the optimisation
  has finished, not that the mean-field approximation captures the posterior exactly.
  Compare marginals against a reference MCMC analysis where feasible, especially for
  divergence times and other strongly correlated parameters.
