#!/usr/bin/env python3
"""Execute the tree normalising flow experiment notebooks non-interactively,
streaming their progress bars live to the terminal.

Like ``examples/run_example.py``, this drives the notebook with an ``nbclient``
client that forwards ``stream`` output (stdout / stderr) to this process as it
arrives, so the ``tqdm`` progress bars of the VI fits and the MCMC reference
chain are visible while the notebook runs rather than only after each cell
finishes.

Every size in ``tree_normalizing_flow.ipynb`` -- optimisation steps, ELBO
samples, and the reference chain's length, burn-in and number of chains -- is
read from an environment variable with the notebook's own value as the default,
so the flags below change what the run does without editing the notebook. A
browser run with nothing set is unchanged.

Examples
--------
A quick smoke run (minutes rather than hours), writing an executed copy
alongside the notebook::

    python experiments/run_tree_flow_experiment.py tree-flow \\
        --num-steps 100 --mcmc-results 500 --mcmc-burnin 200

The full run, overwriting the notebook with its executed outputs::

    python experiments/run_tree_flow_experiment.py tree-flow --inplace

Both experiment notebooks::

    python experiments/run_tree_flow_experiment.py all --inplace
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

_EXPERIMENTS_DIR = Path(__file__).resolve().parent

# Convenience names -> notebook files in this directory.
_KNOWN = {
    "tree-flow": _EXPERIMENTS_DIR / "tree_normalizing_flow.ipynb",
    "boundary-mass": _EXPERIMENTS_DIR / "tree_flow_boundary_mass.ipynb",
}

# Command-line flag -> the environment variable the notebook reads.
_PARAMETERS = {
    "num_steps": "TREEFLOW_TREE_FLOW_NUM_STEPS",
    "learning_rate": "TREEFLOW_TREE_FLOW_LEARNING_RATE",
    "sample_size": "TREEFLOW_TREE_FLOW_SAMPLE_SIZE",
    "elbo_samples": "TREEFLOW_TREE_FLOW_ELBO_SAMPLES",
    "n_samples": "TREEFLOW_TREE_FLOW_N_SAMPLES",
    "mcmc_results": "TREEFLOW_TREE_FLOW_MCMC_RESULTS",
    "mcmc_burnin": "TREEFLOW_TREE_FLOW_MCMC_BURNIN",
    "mcmc_chains": "TREEFLOW_TREE_FLOW_MCMC_CHAINS",
    "mcmc_thin": "TREEFLOW_TREE_FLOW_MCMC_THIN",
    "mcmc_min_ess": "TREEFLOW_TREE_FLOW_MCMC_MIN_ESS",
    "fit_steps": "TREEFLOW_TREE_FLOW_FIT_STEPS",
}


class StreamingNotebookClient(NotebookClient):
    """A ``NotebookClient`` that echoes cell ``stream`` output to this process's
    stdout/stderr live, so progress bars are visible while running (in addition
    to being recorded in the executed notebook)."""

    def output(self, outs, msg, display_id, cell_index):
        if msg.get("header", {}).get("msg_type") == "stream":
            content = msg.get("content", {})
            stream = sys.stderr if content.get("name") == "stderr" else sys.stdout
            stream.write(content.get("text", ""))
            stream.flush()
        return super().output(outs, msg, display_id, cell_index)


def resolve_notebook(name: str) -> Path:
    if name in _KNOWN:
        return _KNOWN[name]
    path = Path(name)
    if path.exists():
        return path.resolve()
    raise SystemExit(
        f"Unknown experiment {name!r}. Choose one of {sorted(_KNOWN)}, 'all', or "
        f"a path to a notebook."
    )


def execute_one(notebook_path: Path, output_path: Path, timeout, kernel: str) -> None:
    nb = nbformat.read(notebook_path, as_version=4)
    client = StreamingNotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel,
        # Run the kernel with the notebook's directory as cwd so its relative
        # paths (the alignment and starting tree) resolve as they do
        # interactively.
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    print(
        f"Executing {notebook_path} (progress streams below; Ctrl-C to stop)\n",
        flush=True,
    )
    try:
        client.execute()
    finally:
        nbformat.write(nb, output_path)
    print(f"\nExecuted notebook written to {output_path}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "notebook",
        help="experiment to run: 'tree-flow', 'boundary-mass', 'all', or a path "
        "to a notebook",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="overwrite each notebook with its executed output (default: write a "
        "'<name>.executed.ipynb' copy alongside it)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output path for a single notebook (ignored for 'all'); overrides --inplace",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="per-cell timeout in seconds (default: no limit -- these runs are long)",
    )
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel name")
    parser.add_argument(
        "--num-steps", type=int, default=None, help="VI optimisation steps per fit"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="VI learning rate"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None, help="MC samples per ELBO gradient"
    )
    parser.add_argument(
        "--elbo-samples", type=int, default=None, help="samples for the final ELBO estimate"
    )
    parser.add_argument(
        "--n-samples", type=int, default=None, help="posterior samples drawn per fit"
    )
    parser.add_argument(
        "--mcmc-results",
        type=int,
        default=None,
        help="reference chain samples kept per chain",
    )
    parser.add_argument(
        "--mcmc-burnin", type=int, default=None, help="reference chain burn-in steps"
    )
    parser.add_argument(
        "--mcmc-chains", type=int, default=None, help="reference chains run in parallel"
    )
    parser.add_argument(
        "--mcmc-thin",
        type=int,
        default=None,
        help="reference chain steps run per sample kept",
    )
    parser.add_argument(
        "--mcmc-min-ess",
        type=float,
        default=None,
        help="effective sample size the reference chain must reach to pass its check",
    )
    parser.add_argument(
        "--fit-steps",
        type=int,
        default=None,
        help="maximum-likelihood steps in the boundary-mass notebook",
    )
    args = parser.parse_args(argv)

    # Injected into the kernel environment; the notebooks read these with their
    # own in-notebook value as the default, so an unset run is unchanged.
    for flag, env_name in _PARAMETERS.items():
        value = getattr(args, flag)
        if value is not None:
            os.environ[env_name] = str(value)

    if args.notebook == "all":
        notebooks = [_KNOWN[name] for name in sorted(_KNOWN)]
        if args.output:
            parser.error("--output cannot be used with 'all'")
    else:
        notebooks = [resolve_notebook(args.notebook)]

    for notebook_path in notebooks:
        if args.output:
            output_path = Path(args.output).resolve()
        elif args.inplace:
            output_path = notebook_path
        else:
            output_path = notebook_path.with_suffix(".executed.ipynb")
        execute_one(notebook_path, output_path, args.timeout, args.kernel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
