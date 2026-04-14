#!/bin/sh
# End-to-end DTA recovery experiment: LPhy simulate -> treeflow_dta_hmc.
#
# Truth (see dta-validation.lphy):
#   K = 4 states
#   pi = [0.4, 0.3, 0.2, 0.1]     (state order: "0","1","2","3")
#   R  = [0.25, 0.15, 0.10, 0.20, 0.10, 0.20]   (row-major upper triangle)
#   mu = 0.2
#   N  = 400 taxa, Yule(lambda=1)

set -eu

DIR="$(cd "$(dirname "$0")" && pwd)"
SEED="${SEED:-42}"
NUM_RESULTS="${NUM_RESULTS:-1000}"
NUM_BURNIN="${NUM_BURNIN:-500}"

: "${LPHY:=$HOME/Git/linguaPhylo/lphy-studio/target/lphy-studio-1.7.0}"
: "${SLPHY:=$HOME/Git/linguaPhylo/bin/slphy}"
: "${TREEFLOW_PY:=$HOME/Git/treeflow/.venv/bin/python}"
: "${TREEFLOW_DTA_HMC:=$HOME/Git/treeflow/.venv/bin/treeflow_dta_hmc}"

cd "$DIR"

echo "[1/3] Simulating with LPhy (seed=$SEED)..."
LPHY="$LPHY" "$SLPHY" dta-validation.lphy -seed "$SEED"

echo "[2/3] Converting nexus -> traits.csv + tree.nwk..."
"$TREEFLOW_PY" convert.py

echo "[3/3] Running treeflow_dta_hmc ($NUM_BURNIN burn-in + $NUM_RESULTS samples)..."
"$TREEFLOW_DTA_HMC" \
  --traits traits.csv \
  --topology tree.nwk \
  --model-file model.yaml \
  --num-results "$NUM_RESULTS" \
  --num-burnin-steps "$NUM_BURNIN" \
  --samples-output samples.csv \
  --seed "$SEED"

echo "Summarising posterior vs truth..."
"$TREEFLOW_PY" summarize.py --samples samples.csv

echo "Done. Samples in $DIR/samples.csv"
