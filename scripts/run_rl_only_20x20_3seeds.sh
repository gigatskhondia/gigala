#!/bin/sh
# Drives the RL-only 20x20 launcher across three seeds (17, 42, 2026) and
# writes every run to its own timestamped sub-directory. Meant to produce
# the reference numbers cited in docs/rl_only_20x20_report.md.
#
# Usage:
#   scripts/run_rl_only_20x20_3seeds.sh                     # 100k-step smoke
#   GEN_RL_RL_TOTAL_TIMESTEPS=3000000 \
#     scripts/run_rl_only_20x20_3seeds.sh                   # full run
#
# Override ``SEEDS`` to sweep a different set:
#   SEEDS="7 11 23" scripts/run_rl_only_20x20_3seeds.sh
#
# Each individual run reuses scripts/run_gen_rl_pipeline_rl_only_20x20.sh,
# so every env-var knob the single-seed launcher exposes is honoured.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

SEEDS=${SEEDS:-"17 42 2026"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-"${REPO_ROOT}/runlogs/gen_rl_rl_only_20x20_3seeds_${TIMESTAMP}"}

mkdir -p "${BASE_OUTPUT_DIR}"

echo "3-seed RL-only 20x20 sweep"
echo "  seeds: ${SEEDS}"
echo "  base output: ${BASE_OUTPUT_DIR}"
echo "  timesteps per seed: ${GEN_RL_RL_TOTAL_TIMESTEPS:-100000}"

for seed in ${SEEDS}; do
  SEED_OUTPUT_DIR="${BASE_OUTPUT_DIR}/seed_${seed}"
  mkdir -p "${SEED_OUTPUT_DIR}"
  echo "============================================================"
  echo "[seed=${seed}] output -> ${SEED_OUTPUT_DIR}"
  GEN_RL_RANDOM_SEED="${seed}" \
    GEN_RL_OUTPUT_DIR="${SEED_OUTPUT_DIR}" \
    "${SCRIPT_DIR}/run_gen_rl_pipeline_rl_only_20x20.sh" \
    "$@" 2>&1 | tee "${SEED_OUTPUT_DIR}/terminal.log"
  echo "[seed=${seed}] done"
done

echo "All seeds complete. Results under ${BASE_OUTPUT_DIR}"
