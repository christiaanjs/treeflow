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
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_OPS=(phylo_likelihood_op node_height_ratio_op conditional_clade_op)

op_output() {
  case "$1" in
    phylo_likelihood_op)   echo "_phylo_likelihood_op.so" ;;
    node_height_ratio_op)  echo "_node_height_ratio_op.so" ;;
    conditional_clade_op)  echo "_conditional_clade_op.so" ;;
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

# Enable AVX2 on x86_64 only. -mavx2 is an x86 flag and g++ rejects it on
# other architectures (e.g. aarch64/arm64), which breaks Docker builds there.
# We deliberately avoid -march=native: enabling AVX512-FP16 trips a bug in the
# Eigen headers bundled with TensorFlow. -O3 with -mavx2 is plenty for these
# kernels on x86; on ARM, -O3 alone already autovectorizes with NEON.
ARCH_FLAGS=()
case "$(uname -m)" in
  x86_64 | amd64) ARCH_FLAGS+=(-mavx2) ;;
esac

for name in "${TARGETS[@]}"; do
  out="$(op_output "${name}")"
  if [ -z "${out}" ]; then
    echo "Unknown op '${name}'. Known: ${ALL_OPS[*]}" >&2
    exit 1
  fi
  SRC="${HERE}/cc/${name}.cc"
  OUT="${HERE}/${out}"
  echo "Building ${OUT}"
  # -I cc so the ops can include the shared tree_traversal.h header.
  "${CXX}" -std=c++17 -shared -fPIC -O3 "${ARCH_FLAGS[@]+"${ARCH_FLAGS[@]}"}" \
    -I"${HERE}/cc" \
    "${SRC}" -o "${OUT}" \
    "${TF_CFLAGS[@]}" "${TF_LFLAGS[@]}"
  echo "Built ${OUT}"
done
