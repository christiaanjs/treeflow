"""Build the native phylogenetic-likelihood TensorFlow custom op.

Usage::

    python -m treeflow.acceleration.native.build

This is a thin Python wrapper around ``build.sh`` so the op can be compiled
without leaving the Python toolchain (e.g. from setup hooks or CI).
"""
import os
import subprocess
import sys


def build() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "build.sh")
    subprocess.run(["bash", script], check=True)
    return os.path.join(here, "_phylo_likelihood_op.so")


if __name__ == "__main__":
    path = build()
    print(f"Built: {path}")
    sys.exit(0)
