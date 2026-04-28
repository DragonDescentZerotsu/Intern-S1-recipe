#!/usr/bin/env bash
set -euo pipefail

SESSION="gemma4"

# 默认 conda 环境名；也可以通过第一个参数覆盖
CONDA_ENV="${1:-vllm}"

MODEL="google/gemma-4-26B-A4B-it"
SERVED_NAME="gemma-4-26B-A4B-it"
CHAT_TEMPLATE="test/chat_templates/tool_chat_template_gemma4.jinja"

BASE_PORT=8001
NUM_GPUS=8

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
else
  CONDA_BASE="$(conda info --base)"
fi

CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Cannot find conda.sh at ${CONDA_SH}"
  echo "Please check your conda installation."
  exit 1
fi

tmux new-session -d -s "${SESSION}" -n "gpu0"

for GPU in $(seq 0 $((NUM_GPUS - 1))); do
  PORT=$((BASE_PORT + GPU))
  WINDOW="gpu${GPU}_p${PORT}"

  if [[ "${GPU}" -eq 0 ]]; then
    tmux rename-window -t "${SESSION}:0" "${WINDOW}"
  else
    tmux new-window -t "${SESSION}" -n "${WINDOW}"
  fi

  CMD="source ${CONDA_SH} && \
conda activate ${CONDA_ENV} && \
CUDA_VISIBLE_DEVICES=${GPU} vllm serve ${MODEL} \
  --host 0.0.0.0 \
  --port ${PORT} \
  --api-key EMPTY \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --served-model-name ${SERVED_NAME} \
  --enable-auto-tool-choice \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --chat-template ${CHAT_TEMPLATE} \
  --language-model-only \
  --async-scheduling \
  2>&1 | tee logs/gemma4_gpu${GPU}_port${PORT}.log"

  tmux send-keys -t "${SESSION}:${WINDOW}" "${CMD}" C-m
done

echo "Started ${NUM_GPUS} Gemma vLLM servers in tmux session: ${SESSION}"
echo "Conda env: ${CONDA_ENV}"
echo
echo "Attach with:"
echo "  tmux attach -t ${SESSION}"
echo
echo "Stop all with:"
echo "  tmux kill-session -t ${SESSION}"
echo
echo "Endpoints:"
for GPU in $(seq 0 $((NUM_GPUS - 1))); do
  echo "  GPU ${GPU}: http://localhost:$((BASE_PORT + GPU))/v1"
done