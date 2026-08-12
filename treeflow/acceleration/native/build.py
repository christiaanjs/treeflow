"""Build treeflow's native TensorFlow custom ops.

Usage::

    python -m treeflow.acceleration.native.build

This is a thin Python wrapper around ``build.sh`` so the ops can be compiled
without leaving the Python toolchain (e.g. from setup hooks or CI). By default
it builds every op; individual ops can be built via :func:`build` (the
phylogenetic likelihood) and :func:`build_node_height_ratio` (the node-height
ratio transform).

Unlike ``build.sh`` invoked directly (which defaults to a portable instruction
set), this wrapper defaults to ``TREEFLOW_NATIVE_ARCH=native``: this entry
point is for building on the machine that will also run the ops (local dev,
`pip install -e .`, the auto-build fallback in test fixtures), so it is safe
to compile for that machine's exact CPU. Set ``TREEFLOW_NATIVE_ARCH=portable``
before calling to opt back into the portable baseline.
"""
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _run_build(*targets: str) -> None:
    script = os.path.join(_HERE, "build.sh")
    env = os.environ.copy()
    env.setdefault("TREEFLOW_NATIVE_ARCH", "native")
    subprocess.run(["bash", script, *targets], check=True, env=env)


def build() -> str:
    """Build the phylogenetic-likelihood op and return its library path."""
    _run_build("phylo_likelihood_op")
    return os.path.join(_HERE, "_phylo_likelihood_op.so")


def build_node_height_ratio() -> str:
    """Build the node-height ratio transform op and return its library path."""
    _run_build("node_height_ratio_op")
    return os.path.join(_HERE, "_node_height_ratio_op.so")


def build_sbn() -> str:
    """Build the SBN topology sampler op and return its library path."""
    _run_build("sbn_op")
    return os.path.join(_HERE, "_sbn_op.so")


def build_all() -> None:
    """Build every native op."""
    _run_build()


if __name__ == "__main__":
    build_all()
    print(f"Built native ops in: {_HERE}")
    sys.exit(0)
