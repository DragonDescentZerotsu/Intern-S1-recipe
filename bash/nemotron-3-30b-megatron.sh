#!/bin/bash
#SBATCH --job-name=nemotron-3-30b-megatron___TDC-scaffold-binary-SFT-wo-herg-c_ToxCast_butkiewicz
#SBATCH --partition=dgx-b200-old-driver
#SBATCH --gpus=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=1-00:00:00
#SBATCH --output=/vast/projects/xia6/apex-gen/tianang/projects/Intern-S1/logs/B200/%x-%j.out
#SBATCH --error=/vast/projects/xia6/apex-gen/tianang/projects/Intern-S1/logs/B200/%x-%j.err

set -euo pipefail

# 你的提交目录（等价于你 srun 里 P="$(pwd)" 的效果）
P="${SLURM_SUBMIT_DIR}"

# mkdir -p "$P/logs"
echo "Node: $SLURM_NODELIST"
echo "Workdir: $P"
echo "Start time: $(date)"

# 在 batch 里用 srun 进入容器并执行命令
srun \
  --container-writable \
  --container-image=docker://nvcr.io/nvidia/nemo:25.11.nemotron_3_nano \
  --container-mounts="$P:$P,$HOME:$HOME,/vast:/vast:rw" \
  --container-workdir="$P" \
  --export=ALL,HOME="$HOME" \
  bash -lc "
    cd '$P'
    nvidia-smi

    # ======== 这里写你要跑的程序 ========
    
    module load cuda/12.8
    conda deactivate
    conda deactivate
    conda deactivate
    cd /vast/projects/xia6/apex-gen/tianang/projects/Intern-S1
    
    torchrun --nproc_per_node 8 train/megatron-bridge/bridge-finetune-nemotron-3-30b.py
  "

echo "End time: $(date)"
