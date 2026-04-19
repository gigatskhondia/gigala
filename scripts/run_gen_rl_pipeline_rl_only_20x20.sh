#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-python3}
PIPELINE_MODE=${GEN_RL_PIPELINE_MODE:-rl_only_exact}
RESOLUTION=${GEN_RL_RESOLUTION:-20}
VOLUME_TARGET=${GEN_RL_VOLUME_TARGET:-0.55}
SOLVER_BACKEND=${GEN_RL_SOLVER_BACKEND:-scipy}
RUNTIME_BUDGET_HOURS=${GEN_RL_RUNTIME_BUDGET_HOURS:-0}
DIRECT_POPULATION=${GEN_RL_DIRECT_POPULATION:-1}
MAX_FULL_EVALS=${GEN_RL_MAX_FULL_EVALS:-4096}
MAX_RL_FULL_EVALS=${GEN_RL_MAX_RL_FULL_EVALS:-200000}
RL_TOTAL_TIMESTEPS=${GEN_RL_RL_TOTAL_TIMESTEPS:-3000000}
RL_N_ENVS=${GEN_RL_RL_N_ENVS:-8}
RL_INFERENCE_ROLLOUTS=${GEN_RL_RL_INFERENCE_ROLLOUTS:-8}
RL_POLICY_SIZE=${GEN_RL_RL_POLICY_SIZE:-large}
RL_VOLUME_SLACK_LOWER=${GEN_RL_RL_VOLUME_SLACK_LOWER:-0.05}
RL_VOLUME_SLACK_UPPER=${GEN_RL_RL_VOLUME_SLACK_UPPER:-0.05}
RL_SKIP_THRESHOLD=${GEN_RL_RL_SKIP_THRESHOLD:-0.95}
RL_SKIP_WARMUP_FRACTION=${GEN_RL_RL_SKIP_WARMUP_FRACTION:-0.3}
RL_HARMONIC_CLAMP=${GEN_RL_RL_HARMONIC_CLAMP:-10.0}
RL_INFEASIBLE_TERMINAL_REWARD=${GEN_RL_RL_INFEASIBLE_TERMINAL_REWARD:--1.0}
MAX_EPISODE_STEPS=${GEN_RL_MAX_EPISODE_STEPS:-800}
RANDOM_SEED=${GEN_RL_RANDOM_SEED:-42}
# Small policy/network on 20x20 is fastest on CPU across Apple Silicon (M1..M4)
# because MPS kernel-dispatch latency dominates tiny forward passes, and
# SubprocVecEnv + MPS interact poorly in forked workers.
RL_DEVICE=${GEN_RL_DEVICE:-cpu}
SPARSE_FLAG=${GEN_RL_RL_SPARSE_REWARD:-1}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${GEN_RL_OUTPUT_DIR:-/tmp/gen_rl_rl_only_20x20_${TIMESTAMP}}
RL_FLAG=""
if [ "${GEN_RL_ENABLE_RL:-1}" = "1" ]; then
  RL_FLAG="--enable-rl"
fi
SPARSE_REWARD_FLAG="--no-rl-sparse-reward"
if [ "${SPARSE_FLAG}" = "1" ]; then
  SPARSE_REWARD_FLAG="--rl-sparse-reward"
fi
export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig_gen_rl}

cd "$REPO_ROOT"

echo "Running RL-only 20x20 gen_rl pipeline from ${REPO_ROOT}"
echo "Default output directory: ${OUTPUT_DIR}"
echo "Defaults: RL enabled, rl_device=${RL_DEVICE}, rl_total_timesteps=${RL_TOTAL_TIMESTEPS}, rl_n_envs=${RL_N_ENVS}, policy_size=${RL_POLICY_SIZE}, sparse_reward=${SPARSE_FLAG}, max_episode_steps=${MAX_EPISODE_STEPS}"

exec "${PYTHON_BIN}" -m gigala.topology.topology_optimiz.gen_rl \
  ${RL_FLAG:+$RL_FLAG} \
  --pipeline-mode "${PIPELINE_MODE}" \
  --rl-device "${RL_DEVICE}" \
  --resolution "${RESOLUTION}" \
  --volume-target "${VOLUME_TARGET}" \
  --solver-backend "${SOLVER_BACKEND}" \
  --runtime-budget-hours "${RUNTIME_BUDGET_HOURS}" \
  --direct-population "${DIRECT_POPULATION}" \
  --rl-total-timesteps "${RL_TOTAL_TIMESTEPS}" \
  --rl-n-envs "${RL_N_ENVS}" \
  --rl-inference-rollouts "${RL_INFERENCE_ROLLOUTS}" \
  --rl-policy-size "${RL_POLICY_SIZE}" \
  --rl-volume-slack-lower "${RL_VOLUME_SLACK_LOWER}" \
  --rl-volume-slack-upper "${RL_VOLUME_SLACK_UPPER}" \
  --rl-skip-threshold "${RL_SKIP_THRESHOLD}" \
  --rl-skip-warmup-fraction "${RL_SKIP_WARMUP_FRACTION}" \
  --rl-harmonic-clamp "${RL_HARMONIC_CLAMP}" \
  --rl-infeasible-terminal-reward "${RL_INFEASIBLE_TERMINAL_REWARD}" \
  --max-episode-steps "${MAX_EPISODE_STEPS}" \
  ${SPARSE_REWARD_FLAG} \
  --max-full-evals "${MAX_FULL_EVALS}" \
  --max-rl-full-evals "${MAX_RL_FULL_EVALS}" \
  --random-seed "${RANDOM_SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
