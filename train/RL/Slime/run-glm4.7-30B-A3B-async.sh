#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# 如果 MLP 平台环境变量未设置，使用本机地址（单节点模式）
# MLP_WORKER_0_HOST: 主节点 IP 地址，用于 Ray 集群和 NCCL 通信
# MLP_SOCKET_IFNAME: 网络接口名称，用于节点间通信
# : ${MLP_WORKER_0_HOST:=$(hostname -I | awk '{print $1}')}
# : ${MLP_SOCKET_IFNAME:=eth0}

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# source "${SCRIPT_DIR}/../scripts/models/glm4.7-30B-A3B.sh"
source "${SCRIPT_DIR}/glm4.7-30B-A3B.sh"  # 我的配置路径

CKPT_ARGS=(
   --hf-checkpoint checkpoints/megatron/hf_version/GLM-4.7-Flash #checkpoints/megatron/hf_version/GLM-4.7-Flash
   --ref-load checkpoints/megatron/megatron_version/GLM-4.7-Flash
)

ROLLOUT_ARGS=(
   --rollout-function-path generate_tdc_async.generate_rollout_fully_async
   --prompt-data DataPrepare/Slime_RL_data/by_task/train/BBB_Martins.jsonl
   --input-key text
   --label-key Y
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle

   --num-rollout 3000
   --rollout-batch-size 36 # 128
   #--over-sampling-batch-size 256
   --n-samples-per-prompt 8
   --rollout-max-response-len 7680
   --rollout-temperature 1.0

   --global-batch-size 256
   #--balance-data
)

# EVAL_ARGS=(
#    --eval-interval 20
#    --eval-prompt-data BBB_Martins_test DataPrepare/Slime_RL_data/by_task/test/BBB_Martins_debug.jsonl
#    --n-samples-per-eval-prompt 2
#    --eval-max-response-len 16384
#    --eval-temperature 0.6
#    --eval-top-p 0.95
# )

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 2
   --expert-tensor-parallel-size 1
   # --decoder-last-pipeline-num-layers 23  # 仅在 pipeline-model-parallel-size > 1 时使用

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group glm4.7-flash
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.9
   # Pin DP settings explicitly. In some sglang/slime combos, dp size can be
   # inferred unexpectedly and cause KV-cache index/value shape mismatches.
   # --sglang-enable-dp-attention
   # --sglang-enable-dp-lm-head
   # --sglang-dp-size 1
   --sglang-moe-dense-tp-size 1

   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4
   
   # In Huggingface page of GLM-4.7-Flash, triton is recommended for attention backend for Blackwell GPUs
   --sglang-attention-backend triton
   --sglang-speculative-draft-attention-backend triton


   --sglang-cuda-graph-max-bs 64
   --sglang-max-running-requests 128
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_tdc_async.generate
   --custom-rm-path reward_tdc.reward_func
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash

   --moe-token-dispatcher-type flex
   --moe-enable-deepep
)

# launch the master node of ray in container
export MASTER_ADDR=${MLP_WORKER_0_HOST:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats  # gpu numbers

# PROJECT_ROOT is 3 levels up from SCRIPT_DIR (train/RL/Slime -> project root)
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:${PROJECT_ROOT}\",
    \"NCCL_IB_DISABLE\": \"1\",
    \"NCCL_CUMEM_ENABLE\": \"0\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NVTE_BWD_LAYERNORM_SM_MARGIN\": \"20\",
    \"NCCL_P2P_LEVEL\": \"NVL\",
    \"TORCH_NCCL_AVOID_RECORD_STREAMS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_MIN_CTAS\": \"4\"
  }
}"

# all the original InfiniteBand (IB) settings are removed as we are only using one node, turn them back on if you have multiple nodes
# Refer to the original InfiniteBand (IB) settings in the original run script in Slime github
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train/RL/Slime/train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 6 \
   --save-debug-rollout-data "data.pt" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
