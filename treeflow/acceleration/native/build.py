"""Build treeflow's native TensorFlow custom ops.

Usage::

    python -m treeflow.acceleration.native.build

This is a thin Python wrapper around ``build.sh`` so the ops can be compiled
without leaving the Python toolchain (e.g. from setup hooks or CI). By default
it builds every op; individual ops can be built via :func:`build` (the
phylogenetic likelihood) and :func:`build_node_height_ratio` (the node-height
ratio transform).
"""
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _run_build(*targets: str) -> None:
    script = os.path.join(_HERE, "build.sh")
    subprocess.run(["bash", script, *targets], check=True)


def build() -> str:
    """Build the phylogenetic-likelihood op and return its library path."""
    _run_build("phylo_likelihood_op")
    return os.path.join(_HERE, "_phylo_likelihood_op.so")


def build_node_height_ratio() -> str:
    """Build the node-height ratio transform op and return its library path."""
    _run_build("node_height_ratio_op")
    return os.path.join(_HERE, "_node_height_ratio_op.so")


def build_all() -> None:
    """Build every native op."""
    _run_build()


if __name__ == "__main__":
    build_all()
    print(f"Built native ops in: {_HERE}")
    sys.exit(0)
