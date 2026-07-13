#!/usr/bin/env python3
"""Execute ``benchmark_pipeline.ipynb`` non-interactively, streaming the sweeps'
progress reporting live to the terminal.

``jupyter nbconvert --execute`` runs the notebook in a kernel subprocess whose
cell output is captured and only surfaced after each cell finishes -- so the
long-running sweep cells appear to hang with no feedback, and ``tqdm.auto``'s
widget bars don't render to a terminal at all. This script instead drives the
notebook with an ``nbclient`` client that forwards ``stream`` outputs (stdout /
stderr) to this process's streams *as they arrive*, and sets
``BENCHMARK_TQDM=text`` so the sweeps use a plain-text ``tqdm`` bar whose
carriage-return updates render as a live single-line bar in the terminal.

Examples
--------
Run with checkpoint resumption (the default -- already-complete configs are
skipped)::

    python experiments/run_benchmark.py

Force a full recompute, writing the executed notebook to a copy::

    python experiments/run_benchmark.py --force --output /tmp/executed.ipynb

Point the checkpoints elsewhere and cap per-cell time to 2 hours::

    python experiments/run_benchmark.py --checkpoint-dir /data/ckpt --timeout 7200
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

_DEFAULT_NOTEBOOK = Path(__file__).resolve().parent / "benchmark_pipeline.ipynb"


class StreamingNotebookClient(NotebookClient):
    """A ``NotebookClient`` that echoes cell ``stream`` output to this process's
    stdout/stderr live, so ``tqdm`` progress bars are visible while the notebook
    runs (in addition to being recorded in the executed notebook)."""

    def output(self, outs, msg, display_id, cell_index):
        if msg.get("header", {}).get("msg_type") == "stream":
            content = msg.get("content", {})
            stream = sys.stderr if content.get("name") == "stderr" else sys.stdout
            stream.write(content.get("text", ""))
            stream.flush()
        return super().output(outs, msg, display_id, cell_index)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "notebook",
        nargs="?",
        default=str(_DEFAULT_NOTEBOOK),
        help="notebook to execute (default: benchmark_pipeline.ipynb next to this script)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="where to write the executed notebook (default: overwrite the input in place)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="per-cell timeout in seconds (default: no limit -- the full sweep can be long)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute every config, ignoring/overwriting the checkpoint cache",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="directory for the resumable per-config checkpoint cache "
        "(default: the notebook's own BENCHMARK_CHECKPOINT_DIR / benchmarks/data/checkpoints)",
    )
    parser.add_argument(
        "--kernel",
        default="python3",
        help="Jupyter kernel name to run the notebook with (default: python3)",
    )
    args = parser.parse_args(argv)

    # Drive the notebook's config cell + the sweeps' progress flavour via the
    # environment, which nbclient passes through to the kernel subprocess.
    os.environ["BENCHMARK_TQDM"] = "text"
    if args.force:
        os.environ["BENCHMARK_FORCE"] = "1"
    if args.checkpoint_dir is not None:
        os.environ["BENCHMARK_CHECKPOINT_DIR"] = args.checkpoint_dir

    notebook_path = Path(args.notebook).resolve()
    output_path = Path(args.output).resolve() if args.output else notebook_path

    nb = nbformat.read(notebook_path, as_version=4)
    client = StreamingNotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name=args.kernel,
        # Run the kernel with the notebook's directory as cwd so its relative
        # paths (benchmarks/, benchmarks/data/...) resolve exactly as they do
        # under `jupyter nbconvert --inplace`.
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )

    print(f"Executing {notebook_path} (progress streams below; Ctrl-C to stop)\n", flush=True)
    try:
        client.execute()
    finally:
        nbformat.write(nb, output_path)
    print(f"\nExecuted notebook written to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
