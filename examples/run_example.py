#!/usr/bin/env python3
"""Execute the example notebooks non-interactively, streaming their progress
bars live to the terminal.

Like ``experiments/run_benchmark.py``, this drives the notebook with an
``nbclient`` client that forwards ``stream`` output (stdout / stderr) to this
process as it arrives, so ``tqdm`` progress bars are visible while the notebook
runs instead of only after each cell finishes (which is what
``jupyter nbconvert --execute`` gives you). The example notebooks use the text
``tqdm`` (not the ``tqdm.notebook`` widget), so their bars stream as plain text.

Examples
--------
Run one example, streaming progress, writing the executed copy alongside::

    python examples/run_example.py carnivores

Run both examples in place (overwriting them with their executed outputs)::

    python examples/run_example.py all --inplace

Run a notebook by path with a 4-hour per-cell timeout::

    python examples/run_example.py examples/rates-and-dates.ipynb --timeout 14400
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

_EXAMPLES_DIR = Path(__file__).resolve().parent

# Convenience names -> notebook files in this directory.
_KNOWN = {
    "carnivores": _EXAMPLES_DIR / "carnivores.ipynb",
    "rates-and-dates": _EXAMPLES_DIR / "rates-and-dates.ipynb",
}


class StreamingNotebookClient(NotebookClient):
    """A ``NotebookClient`` that echoes cell ``stream`` output to this process's
    stdout/stderr live, so ``tqdm`` progress bars are visible while running (in
    addition to being recorded in the executed notebook)."""

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
        f"Unknown example {name!r}. Choose one of {sorted(_KNOWN)}, 'all', or a "
        f"path to a notebook."
    )


def execute_one(notebook_path: Path, output_path: Path, timeout, kernel: str) -> None:
    nb = nbformat.read(notebook_path, as_version=4)
    client = StreamingNotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel,
        # Run the kernel with the notebook's directory as cwd so its relative
        # paths (data files, model yaml) resolve as they do interactively.
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    print(f"Executing {notebook_path} (progress streams below; Ctrl-C to stop)\n", flush=True)
    try:
        client.execute()
    finally:
        nbformat.write(nb, output_path)
    print(f"\nExecuted notebook written to {output_path}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "notebook",
        help="example to run: 'carnivores', 'rates-and-dates', 'all', or a path to a notebook",
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
        help="per-cell timeout in seconds (default: no limit -- VI runs can be long)",
    )
    parser.add_argument(
        "--kernel",
        default="python3",
        help="Jupyter kernel name (default: python3)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="VI optimisation iterations (carnivores; sets TREEFLOW_EXAMPLE_NUM_STEPS). "
        "Default: the notebook's own value.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help="number of independent VI runs (carnivores; sets TREEFLOW_EXAMPLE_N_RUNS). "
        "Default: the notebook's own value.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="posterior samples drawn per run (carnivores; sets TREEFLOW_EXAMPLE_N_SAMPLES). "
        "Default: the notebook's own value.",
    )
    args = parser.parse_args(argv)

    # Injected into the kernel environment; the example notebooks read these with
    # a default of their in-notebook value, so a browser run with nothing set is
    # unchanged. Handy for a quick smoke run, e.g. --num-steps 200 --n-runs 2.
    for value, env_name in (
        (args.num_steps, "TREEFLOW_EXAMPLE_NUM_STEPS"),
        (args.n_runs, "TREEFLOW_EXAMPLE_N_RUNS"),
        (args.n_samples, "TREEFLOW_EXAMPLE_N_SAMPLES"),
    ):
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
