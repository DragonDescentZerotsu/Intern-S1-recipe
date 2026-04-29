#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SESSION="gpt20b"
BASE_PORT="${VLLM_BASE_PORT:-8001}"
NUM_SERVERS="${VLLM_NUM_SERVERS:-4}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
API_BASE="${TEST_API_BASE:-http://127.0.0.1:9002/v1}"
WAIT_TIMEOUT_SECONDS="${VLLM_WAIT_TIMEOUT_SECONDS:-3600}"
WAIT_INTERVAL_SECONDS="${VLLM_WAIT_INTERVAL_SECONDS:-10}"
GPU_IDS="${VLLM_GPU_IDS:-0,1,2,3}"
GPU_WAIT_TIMEOUT_SECONDS="${VLLM_GPU_WAIT_TIMEOUT_SECONDS:-1800}"
GPU_WAIT_INTERVAL_SECONDS="${VLLM_GPU_WAIT_INTERVAL_SECONDS:-5}"
VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-reasonv}"
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-reasonv}"
VLLM_SESSION_ACTIVE=0

if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
else
  CONDA_BASE="$(conda info --base)"
fi

CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Cannot find conda.sh at ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"

cleanup_vllm() {
  if [[ "${VLLM_SESSION_ACTIVE}" != "1" ]]; then
    return
  fi

  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux kill-session -t "${SESSION}"
  fi
  VLLM_SESSION_ACTIVE=0

  bash ./bash/wait_for_gpus_free.sh \
    --gpus "${GPU_IDS}" \
    --timeout-seconds "${GPU_WAIT_TIMEOUT_SECONDS}" \
    --interval-seconds "${GPU_WAIT_INTERVAL_SECONDS}"
}

start_and_wait() {
  local starter_script="$1"
  local served_model="$2"

  bash "${starter_script}" "${VLLM_CONDA_ENV}"
  VLLM_SESSION_ACTIVE=1
  bash ./bash/wait_for_vllm_servers.sh \
    --base-port "${BASE_PORT}" \
    --num-servers "${NUM_SERVERS}" \
    --api-key "${API_KEY}" \
    --model "${served_model}" \
    --timeout-seconds "${WAIT_TIMEOUT_SECONDS}" \
    --interval-seconds "${WAIT_INTERVAL_SECONDS}"
}

run_eval() {
  local served_model="$1"
  local log_file_name="$2"
  shift 2

  conda activate "${EVAL_CONDA_ENV}"

  python test/test_tdc_via_api_F1_TRIM.py \
    --provider local \
    --api-base "${API_BASE}" \
    --api-key "${API_KEY}" \
    --model "${served_model}" \
    --data-dir DataPrepare/TDC_no_conflict_labels_salt_removed/test \
    --tool-mode similar \
    --max-tokens 20480 \
    --chat-template-kwargs-json '{"enable_thinking": true}' \
    --num-processes 64 \
    --log-file \
    --log-file-name "${log_file_name}" \
    --save-reasoning-trajectories \
    "$@"
}

run_model() {
  local starter_script="$1"
  local served_model="$2"

  start_and_wait "${starter_script}" "${served_model}"

  run_eval \
    "${served_model}" \
    "test_${served_model}-similar-tool_{t_stamp}.log"

  run_eval \
    "${served_model}" \
    "test_${served_model}-similar-tool-neighbors_only_{t_stamp}.log" \
    --similar-tool-feature-view neighbors_only

  cleanup_vllm
}

trap cleanup_vllm EXIT

run_model ./bash/vllm_starters/vllm_starter_gpt-20b-RL-baseline.sh gpt-oss-20b-RL-baseline
run_model ./bash/vllm_starters/vllm_starter_gpt-20b-SFT.sh gpt-oss-20b-SFT
run_model ./bash/vllm_starters/vllm_starter_gpt-20b-SFT+RL.sh gpt-oss-20b-SFT+RL

trap - EXIT
