#!/bin/bash
#SBATCH --job-name=Slime-RL-GLM-4.7-Flash-length-penalty-w-tool-reward
#SBATCH --partition=dgx-b200
#SBATCH --gpus=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=03:00:00
#SBATCH --output=/vast/projects/xia6/apex-gen/tianang/projects/Intern-S1/logs/B200/%x-%j.out
#SBATCH --error=/vast/projects/xia6/apex-gen/tianang/projects/Intern-S1/logs/B200/%x-%j.err

set -euo pipefail

# 你的提交目录（等价于你 srun 里 P="$(pwd)" 的效果）
P="${SLURM_SUBMIT_DIR}"

# mkdir -p "$P/logs"
echo "Node: $SLURM_NODELIST"
echo "Workdir: $P"
echo "Start time: $(date)"

CACHE=/vast/projects/xia6/apex-gen/tianang/container_cache
export XDG_RUNTIME_DIR=$CACHE/xdg_runtime
export ENROOT_CACHE_PATH=$CACHE/enroot_cache
export ENROOT_DATA_PATH=$CACHE/enroot_data
export ENROOT_TEMP_PATH=$CACHE/enroot_tmp

# 在 batch 里用 srun 进入容器并执行命令
srun \
  --container-writable \
  --container-image=$CACHE/slime_latest.sqsh \
  --container-mounts="$P:$P,$HOME:$HOME,/vast:/vast:rw" \
  --container-workdir="$P" \
  --export=ALL,HOME="$HOME" \
  bash -lc "
    cd '$P'
    nvidia-smi

    # ======== 这里写你要跑的程序 ========
    
    module load cuda
    conda deactivate
    conda deactivate
    conda deactivate
    cd /vast/projects/xia6/apex-gen/tianang/projects/Intern-S1

    bash bash/Slime_docker_update.sh
    
    bash train/RL/Slime/run-glm4.7-30B-A3B-async.sh
  "

echo "End time: $(date)"
