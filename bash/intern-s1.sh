#!/bin/bash
#SBATCH --job-name=intern-s1
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# 可选：创建日志目录（若不存在）
mkdir -p logs

# 可选：加载环境/激活虚拟环境（按你们环境需要修改）
# module load cuda/xxx
# source /path/to/conda.sh
# conda activate your-env

# 建议限制线程数，避免 Python/Numpy 过度并行
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# 进入项目目录并运行
cd /vast/projects/xia6/apex-gen/tianang/projects/Intern-S1

# 用 srun 启动，确保资源绑定到该作业步
srun python intern-vllm-A100-CoT-multi-sample.py
