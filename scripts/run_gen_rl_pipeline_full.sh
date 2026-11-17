#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-python3}
PIPELINE_MODE=${GEN_RL_PIPELINE_MODE:-multistage}
RESOLUTION=${GEN_RL_RESOLUTION:-64}
VOLUME_TARGET=${GEN_RL_VOLUME_TARGET:-0.55}
SOLVER_BACKEND=${GEN_RL_SOLVER_BACKEND:-scipy}
RUNTIME_BUDGET_HOURS=${GEN_RL_RUNTIME_BUDGET_HOURS:-3.0}
COARSE_POPULATION=${GEN_RL_COARSE_POPULATION:-128}
COARSE_GENERATIONS=${GEN_RL_COARSE_GENERATIONS:-250}
COARSE_ELITE_COUNT=${GEN_RL_COARSE_ELITE_COUNT:-16}
STAGE32_TOP_K=${GEN_RL_STAGE32_TOP_K:-8}
STAGE64_TOP_K=${GEN_RL_STAGE64_TOP_K:-2}
LOCAL_SEARCH_STEPS32=${GEN_RL_LOCAL_SEARCH_STEPS32:-48}
LOCAL_SEARCH_STEPS64=${GEN_RL_LOCAL_SEARCH_STEPS64:-96}
RL_TOTAL_TIMESTEPS=${GEN_RL_RL_TOTAL_TIMESTEPS:-100000}
MAX_FULL_EVALS=${GEN_RL_MAX_FULL_EVALS:-20000}
MAX_RL_FULL_EVALS=${GEN_RL_MAX_RL_FULL_EVALS:-5000}
RANDOM_SEED=${GEN_RL_RANDOM_SEED:-42}
RL_DEVICE=${GEN_RL_DEVICE:-auto}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${GEN_RL_OUTPUT_DIR:-/tmp/gen_rl_cli_full_${RESOLUTION}_${TIMESTAMP}}
RL_FLAG=""
if [ "${GEN_RL_ENABLE_RL:-1}" = "1" ]; then
  RL_FLAG="--enable-rl"
fi
export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig_gen_rl}

cd "$REPO_ROOT"

echo "Running full gen_rl pipeline from ${REPO_ROOT}"
echo "Default output directory: ${OUTPUT_DIR}"

exec "${PYTHON_BIN}" -m gigala.topology.topology_optimiz.gen_rl \
  ${RL_FLAG:+$RL_FLAG} \
  --pipeline-mode "${PIPELINE_MODE}" \
  --rl-device "${RL_DEVICE}" \
  --resolution "${RESOLUTION}" \
  --volume-target "${VOLUME_TARGET}" \
  --solver-backend "${SOLVER_BACKEND}" \
  --runtime-budget-hours "${RUNTIME_BUDGET_HOURS}" \
  --coarse-population "${COARSE_POPULATION}" \
  --coarse-generations "${COARSE_GENERATIONS}" \
  --coarse-elite-count "${COARSE_ELITE_COUNT}" \
  --stage32-top-k "${STAGE32_TOP_K}" \
  --stage64-top-k "${STAGE64_TOP_K}" \
  --local-search-steps32 "${LOCAL_SEARCH_STEPS32}" \
  --local-search-steps64 "${LOCAL_SEARCH_STEPS64}" \
  --rl-total-timesteps "${RL_TOTAL_TIMESTEPS}" \
  --max-full-evals "${MAX_FULL_EVALS}" \
  --max-rl-full-evals "${MAX_RL_FULL_EVALS}" \
  --random-seed "${RANDOM_SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
