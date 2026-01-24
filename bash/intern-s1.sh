#!/bin/bash
#SBATCH --job-name=intern-s1-mini___TDC-scaffold-binary-SFT-wo-herg-c_ToxCast_butkiewicz
#SBATCH --partition=dgx-b200-old-driver
#SBATCH --gpus=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/vast/projects/xia6/apex-gen/tianang/projects/Intern-S1/logs/B200/%x-%j.out
#SBATCH --error=/vast/projects/xia6/apex-gen/tianang/projects/Intern-S1/logs/B200/%x-%j.err

set -euo pipefail

# 可选：创建日志目录（若不存在）
#mkdir -p logs

# 可选：加载环境/激活虚拟环境（按你们环境需要修改）
module load cuda/12.8
source /vast/projects/xia6/apex-gen/tianang/miniconda3/etc/profile.d/conda.sh
conda activate llamafactory

# 建议限制线程数，避免 Python/Numpy 过度并行
#export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
#export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo $CUDA_HOME
nvidia-smi

# srun --gpus=1 python - << 'PY'
# import torch
# print("torch.cuda.is_available():", torch.cuda.is_available())
# print("device_count:", torch.cuda.device_count())
# PY

# 进入项目目录并运行
cd /vast/projects/xia6/apex-gen/tianang/projects/Intern-S1

# 用 srun 启动，确保资源绑定到该作业步
#srun python intern-vllm-A100-CoT-multi-sample.py
# srun --gpus=8 --ntasks=1 --cpus-per-task=32 llamafactory-cli train train_config/llamafactory-intern-s1-mini-sft.yaml
llamafactory-cli train train/train_config/llamafactory-TDC-SFT-binary-wo-herg-c_ToxCast_butkiewicz-intern-s1-mini.yaml