#!/usr/bin/env bash
# Build the native phylogenetic-likelihood TensorFlow custom op.
#
# Compiles cc/phylo_likelihood_op.cc into a shared library that TensorFlow can
# load via tf.load_op_library. Uses the compile/link flags reported by the
# installed TensorFlow so the C++ ABI matches the running runtime.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${HERE}/cc/phylo_likelihood_op.cc"
OUT="${HERE}/_phylo_likelihood_op.so"

CXX="${CXX:-g++}"

read -r -a TF_CFLAGS <<<"$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')"
read -r -a TF_LFLAGS <<<"$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')"

# Enable AVX2 on x86_64 only. -mavx2 is an x86 flag and g++ rejects it on
# other architectures (e.g. aarch64/arm64), which breaks Docker builds there.
# We deliberately avoid -march=native: enabling AVX512-FP16 trips a bug in the
# Eigen headers bundled with TensorFlow. -O3 with -mavx2 is plenty for this
# kernel on x86; on ARM, -O3 alone already autovectorizes with NEON.
ARCH_FLAGS=()
case "$(uname -m)" in
  x86_64 | amd64) ARCH_FLAGS+=(-mavx2) ;;
esac

echo "Building ${OUT}"
"${CXX}" -std=c++17 -shared -fPIC -O3 "${ARCH_FLAGS[@]+"${ARCH_FLAGS[@]}"}" \
  "${SRC}" -o "${OUT}" \
  "${TF_CFLAGS[@]}" "${TF_LFLAGS[@]}"

echo "Built ${OUT}"
