#!/usr/bin/env bash
# Build treeflow's native TensorFlow custom ops.
#
# Compiles each cc/<op>.cc into a shared library that TensorFlow can load via
# tf.load_op_library. Uses the compile/link flags reported by the installed
# TensorFlow so the C++ ABI matches the running runtime.
#
# Usage:
#   build.sh                      # build every op
#   build.sh phylo_likelihood_op  # build just the named op (basename, no .cc)
#
# Instruction-set baseline (see README.md#instruction-set-baseline):
#   TREEFLOW_NATIVE_ARCH=portable (default) — a baseline shared by every
#     supported x86_64 host (-mavx2; no flag, i.e. the compiler default, on
#     other architectures). Required whenever the .so may run on a different
#     machine than the one that compiled it, e.g. a published Docker image.
#   TREEFLOW_NATIVE_ARCH=native — -march=native, tuned for the exact CPU
#     compiling it. Only safe when build and run happen on the same machine.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_OPS=(phylo_likelihood_op node_height_ratio_op tree_affine_op)

op_output() {
  case "$1" in
    phylo_likelihood_op)   echo "_phylo_likelihood_op.so" ;;
    node_height_ratio_op)  echo "_node_height_ratio_op.so" ;;
    tree_affine_op)        echo "_tree_affine_op.so" ;;
    *) echo "" ;;
  esac
}

if [ "$#" -gt 0 ]; then
  TARGETS=("$@")
else
  TARGETS=("${ALL_OPS[@]}")
fi

CXX="${CXX:-g++}"

read -r -a TF_CFLAGS <<<"$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')"
read -r -a TF_LFLAGS <<<"$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')"

NATIVE_ARCH="${TREEFLOW_NATIVE_ARCH:-portable}"
case "${NATIVE_ARCH}" in
  native)
    ARCH_FLAGS=(-march=native)
    ;;
  portable)
    ARCH_FLAGS=()
    case "$(uname -m)" in
      x86_64 | amd64) ARCH_FLAGS+=(-mavx2) ;;
    esac
    ;;
  *)
    echo "Unknown TREEFLOW_NATIVE_ARCH '${NATIVE_ARCH}'. Expected 'portable' or 'native'." >&2
    exit 1
    ;;
esac

for name in "${TARGETS[@]}"; do
  out="$(op_output "${name}")"
  if [ -z "${out}" ]; then
    echo "Unknown op '${name}'. Known: ${ALL_OPS[*]}" >&2
    exit 1
  fi
  SRC="${HERE}/cc/${name}.cc"
  OUT="${HERE}/${out}"
  echo "Building ${OUT} (${NATIVE_ARCH})"
  # -I cc so the ops can include the shared tree_traversal.h header.
  "${CXX}" -std=c++17 -shared -fPIC -O3 "${ARCH_FLAGS[@]+"${ARCH_FLAGS[@]}"}" \
    -I"${HERE}/cc" \
    "${SRC}" -o "${OUT}" \
    "${TF_CFLAGS[@]}" "${TF_LFLAGS[@]}"
  echo "Built ${OUT}"
done
