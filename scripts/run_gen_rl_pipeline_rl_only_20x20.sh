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
RL_TOTAL_TIMESTEPS=${GEN_RL_RL_TOTAL_TIMESTEPS:-100000}
RL_N_ENVS=${GEN_RL_RL_N_ENVS:-8}
RL_INFERENCE_ROLLOUTS=${GEN_RL_RL_INFERENCE_ROLLOUTS:-8}
RL_POLICY_SIZE=${GEN_RL_RL_POLICY_SIZE:-large}
RL_VOLUME_SLACK_LOWER=${GEN_RL_RL_VOLUME_SLACK_LOWER:-0.05}
RL_VOLUME_SLACK_UPPER=${GEN_RL_RL_VOLUME_SLACK_UPPER:-0.05}
RL_SKIP_THRESHOLD=${GEN_RL_RL_SKIP_THRESHOLD:-0.95}
RL_SKIP_WARMUP_FRACTION=${GEN_RL_RL_SKIP_WARMUP_FRACTION:-0.3}
RL_HARMONIC_CLAMP=${GEN_RL_RL_HARMONIC_CLAMP:-10.0}
RL_INFEASIBLE_TERMINAL_REWARD=${GEN_RL_RL_INFEASIBLE_TERMINAL_REWARD:--1.0}
RL_ENT_COEF=${GEN_RL_RL_ENT_COEF:-0.03}
RL_ENT_COEF_FINAL=${GEN_RL_RL_ENT_COEF_FINAL:-0.005}
RL_TARGET_KL=${GEN_RL_RL_TARGET_KL:-0.03}
RL_TARGET_KL_FINAL=${GEN_RL_RL_TARGET_KL_FINAL:-0.08}
RL_LR_INITIAL=${GEN_RL_RL_LR_INITIAL:-3e-4}
RL_LR_FINAL=${GEN_RL_RL_LR_FINAL:-5e-5}
RL_LR_SCHEDULE=${GEN_RL_RL_LR_SCHEDULE:-cosine}
RL_BATCH_SIZE=${GEN_RL_RL_BATCH_SIZE:-256}
RL_BEST_HARVEST_TOPK=${GEN_RL_RL_BEST_HARVEST_TOPK:-4}
RL_VOLUME_TOLERANCE=${GEN_RL_RL_VOLUME_TOLERANCE:-0.03}
RL_SEED_STRATEGY=${GEN_RL_RL_SEED_STRATEGY:-random_near_target}
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
echo "Defaults: RL enabled, rl_device=${RL_DEVICE}, rl_total_timesteps=${RL_TOTAL_TIMESTEPS}, rl_n_envs=${RL_N_ENVS}, policy_size=${RL_POLICY_SIZE}, sparse_reward=${SPARSE_FLAG}, max_episode_steps=${MAX_EPISODE_STEPS}, ent_coef=${RL_ENT_COEF}->${RL_ENT_COEF_FINAL}, target_kl=${RL_TARGET_KL}->${RL_TARGET_KL_FINAL}, lr=${RL_LR_INITIAL}->${RL_LR_FINAL} (${RL_LR_SCHEDULE}), batch_size=${RL_BATCH_SIZE}, best_harvest_topk=${RL_BEST_HARVEST_TOPK}, rl_volume_tolerance=${RL_VOLUME_TOLERANCE}, rl_seed_strategy=${RL_SEED_STRATEGY}"

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
  --rl-ent-coef "${RL_ENT_COEF}" \
  --rl-ent-coef-final "${RL_ENT_COEF_FINAL}" \
  --rl-target-kl "${RL_TARGET_KL}" \
  --rl-target-kl-final "${RL_TARGET_KL_FINAL}" \
  --rl-lr-initial "${RL_LR_INITIAL}" \
  --rl-lr-final "${RL_LR_FINAL}" \
  --rl-lr-schedule "${RL_LR_SCHEDULE}" \
  --rl-batch-size "${RL_BATCH_SIZE}" \
  --rl-best-harvest-topk "${RL_BEST_HARVEST_TOPK}" \
  --rl-volume-tolerance "${RL_VOLUME_TOLERANCE}" \
  --rl-seed-strategy "${RL_SEED_STRATEGY}" \
  --max-episode-steps "${MAX_EPISODE_STEPS}" \
  ${SPARSE_REWARD_FLAG} \
  --max-full-evals "${MAX_FULL_EVALS}" \
  --max-rl-full-evals "${MAX_RL_FULL_EVALS}" \
  --random-seed "${RANDOM_SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
