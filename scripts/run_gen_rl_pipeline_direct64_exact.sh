#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-python3}
PIPELINE_MODE=${GEN_RL_PIPELINE_MODE:-direct64_exact}
RESOLUTION=${GEN_RL_RESOLUTION:-64}
VOLUME_TARGET=${GEN_RL_VOLUME_TARGET:-0.55}
SOLVER_BACKEND=${GEN_RL_SOLVER_BACKEND:-scipy}
RUNTIME_BUDGET_HOURS=${GEN_RL_RUNTIME_BUDGET_HOURS:-0.05}
DIRECT_POPULATION=${GEN_RL_DIRECT_POPULATION:-12}
DIRECT_ELITE_COUNT=${GEN_RL_DIRECT_ELITE_COUNT:-4}
DIRECT_OFFSPRING_BATCH=${GEN_RL_DIRECT_OFFSPRING_BATCH:-6}
DIRECT_ARCHIVE_SIZE=${GEN_RL_DIRECT_ARCHIVE_SIZE:-8}
DIRECT_RESTART_STAGNATION_EVALS=${GEN_RL_DIRECT_RESTART_STAGNATION_EVALS:-48}
MAX_FULL_EVALS=${GEN_RL_MAX_FULL_EVALS:-120}
MAX_RL_FULL_EVALS=${GEN_RL_MAX_RL_FULL_EVALS:-128}
RL_TOTAL_TIMESTEPS=${GEN_RL_RL_TOTAL_TIMESTEPS:-4096}
RL_ARCHIVE_TOP_K=${GEN_RL_RL_ARCHIVE_TOP_K:-3}
RANDOM_SEED=${GEN_RL_RANDOM_SEED:-42}
RL_DEVICE=${GEN_RL_DEVICE:-auto}
WORKERS=${GEN_RL_WORKERS:-auto}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${GEN_RL_OUTPUT_DIR:-/tmp/gen_rl_direct64_exact_${TIMESTAMP}}
RL_FLAG=""
if [ "${GEN_RL_ENABLE_RL:-0}" = "1" ]; then
  RL_FLAG="--enable-rl"
fi
export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig_gen_rl}

cd "$REPO_ROOT"

echo "Running direct64 exact gen_rl pipeline from ${REPO_ROOT}"
echo "Default output directory: ${OUTPUT_DIR}"

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
  --rl-archive-top-k "${RL_ARCHIVE_TOP_K}" \
  --max-full-evals "${MAX_FULL_EVALS}" \
  --max-rl-full-evals "${MAX_RL_FULL_EVALS}" \
  --random-seed "${RANDOM_SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
