#!/usr/bin/env bash
set -euo pipefail

SESSION="gpt120b"

# 默认 conda 环境名；也可以通过第一个参数覆盖
CONDA_ENV="${1:-vllm}"

MODEL="openai/gpt-oss-120b"
SERVED_NAME="gpt-oss-120b"

BASE_PORT=8001
NUM_INSTANCES=4
GPUS_PER_INSTANCE=2
TOTAL_GPUS=$((NUM_INSTANCES * GPUS_PER_INSTANCE))

mkdir -p logs

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session '${SESSION}' already exists."
  echo "Kill it first with:"
  echo "  tmux kill-session -t ${SESSION}"
  exit 1
fi

# 找 conda 初始化脚本
if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -x "/data1/tianang/anaconda3/condabin/conda" ]]; then
  CONDA_BASE="$(/data1/tianang/anaconda3/condabin/conda info --base)"
else
  echo "Cannot find conda. Please make sure conda is available on PATH."
  exit 1
fi

CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Cannot find conda.sh at ${CONDA_SH}"
  echo "Please check your conda installation."
  exit 1
fi

tmux new-session -d -s "${SESSION}" -n "gpu0-1"

for INSTANCE in $(seq 0 $((NUM_INSTANCES - 1))); do
  GPU_START=$((INSTANCE * GPUS_PER_INSTANCE))
  GPU_END=$((GPU_START + GPUS_PER_INSTANCE - 1))
  GPU_LIST="${GPU_START},${GPU_END}"
  PORT=$((BASE_PORT + INSTANCE))
  WINDOW="gpu${GPU_START}-${GPU_END}_p${PORT}"

  if [[ "${INSTANCE}" -eq 0 ]]; then
    tmux rename-window -t "${SESSION}:0" "${WINDOW}"
  else
    tmux new-window -t "${SESSION}" -n "${WINDOW}"
  fi

  CMD="source ${CONDA_SH} && \
conda activate ${CONDA_ENV} && \
CUDA_VISIBLE_DEVICES=${GPU_LIST} vllm serve ${MODEL} \
  --host 0.0.0.0 \
  --port ${PORT} \
  --api-key EMPTY \
  --tensor-parallel-size ${GPUS_PER_INSTANCE} \
  --tool-call-parser openai \
  --enable-auto-tool-choice \
  --served-model-name ${SERVED_NAME} \
  2>&1 | tee logs/gpt-oss-120b_gpu${GPU_START}-${GPU_END}_port${PORT}.log"

  tmux send-keys -t "${SESSION}:${WINDOW}" "${CMD}" C-m
done

echo "Started ${NUM_INSTANCES} GPT-OSS 120B vLLM servers across ${TOTAL_GPUS} GPUs in tmux session: ${SESSION}"
echo "Conda env: ${CONDA_ENV}"
echo
echo "Attach with:"
echo "  tmux attach -t ${SESSION}"
echo
echo "Stop all with:"
echo "  tmux kill-session -t ${SESSION}"
echo
echo "Endpoints:"
for INSTANCE in $(seq 0 $((NUM_INSTANCES - 1))); do
  GPU_START=$((INSTANCE * GPUS_PER_INSTANCE))
  GPU_END=$((GPU_START + GPUS_PER_INSTANCE - 1))
  echo "  GPUs ${GPU_START},${GPU_END}: http://localhost:$((BASE_PORT + INSTANCE))/v1"
done
