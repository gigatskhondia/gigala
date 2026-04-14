#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-python3}
PIPELINE_MODE=${GEN_RL_PIPELINE_MODE:-direct64_exact}
RESOLUTION=${GEN_RL_RESOLUTION:-64}
VOLUME_TARGET=${GEN_RL_VOLUME_TARGET:-0.55}
SOLVER_BACKEND=${GEN_RL_SOLVER_BACKEND:-scipy}
RUNTIME_BUDGET_HOURS=${GEN_RL_RUNTIME_BUDGET_HOURS:-0}
DIRECT_POPULATION=${GEN_RL_DIRECT_POPULATION:-48}
DIRECT_ELITE_COUNT=${GEN_RL_DIRECT_ELITE_COUNT:-8}
DIRECT_OFFSPRING_BATCH=${GEN_RL_DIRECT_OFFSPRING_BATCH:-16}
DIRECT_ARCHIVE_SIZE=${GEN_RL_DIRECT_ARCHIVE_SIZE:-32}
DIRECT_RESTART_STAGNATION_EVALS=${GEN_RL_DIRECT_RESTART_STAGNATION_EVALS:-400}
MAX_FULL_EVALS=${GEN_RL_MAX_FULL_EVALS:-20000}
MAX_RL_FULL_EVALS=${GEN_RL_MAX_RL_FULL_EVALS:-5000}
RL_TOTAL_TIMESTEPS=${GEN_RL_RL_TOTAL_TIMESTEPS:-100000}
RANDOM_SEED=${GEN_RL_RANDOM_SEED:-42}
RL_DEVICE=${GEN_RL_DEVICE:-mps}
WORKERS=${GEN_RL_WORKERS:-auto}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${GEN_RL_OUTPUT_DIR:-/tmp/gen_rl_direct64_exact_full_${TIMESTAMP}}
RL_FLAG=""
if [ "${GEN_RL_ENABLE_RL:-1}" = "1" ]; then
  RL_FLAG="--enable-rl"
fi
export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig_gen_rl}

cd "$REPO_ROOT"

echo "Running full direct64 exact gen_rl pipeline from ${REPO_ROOT}"
echo "Default output directory: ${OUTPUT_DIR}"
echo "Defaults: RL enabled, rl_device=${RL_DEVICE}, runtime_budget_hours=${RUNTIME_BUDGET_HOURS}, max_full_evals=${MAX_FULL_EVALS}, max_rl_full_evals=${MAX_RL_FULL_EVALS}"

exec "${PYTHON_BIN}" -m gigala.topology.topology_optimiz.gen_rl \
  ${RL_FLAG:+$RL_FLAG} \
  --pipeline-mode "${PIPELINE_MODE}" \
  --rl-device "${RL_DEVICE}" \
  --resolution "${RESOLUTION}" \
  --volume-target "${VOLUME_TARGET}" \
  --solver-backend "${SOLVER_BACKEND}" \
  --runtime-budget-hours "${RUNTIME_BUDGET_HOURS}" \
  --direct-population "${DIRECT_POPULATION}" \
  --direct-elite-count "${DIRECT_ELITE_COUNT}" \
  --direct-offspring-batch "${DIRECT_OFFSPRING_BATCH}" \
  --direct-archive-size "${DIRECT_ARCHIVE_SIZE}" \
  --direct-restart-stagnation-evals "${DIRECT_RESTART_STAGNATION_EVALS}" \
  --workers "${WORKERS}" \
  --rl-total-timesteps "${RL_TOTAL_TIMESTEPS}" \
  --max-full-evals "${MAX_FULL_EVALS}" \
  --max-rl-full-evals "${MAX_RL_FULL_EVALS}" \
  --random-seed "${RANDOM_SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
