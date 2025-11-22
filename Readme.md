# This repo is for Intern-S1 recipe experiment

## 1. Pull my latest modified NeMo docker image
### On slurm cluster, use:
```bash
srun --partition=dgx-b200 \
     --pty \
     --nodes=1 \
     --gpus=8 \
     --cpus-per-task=64 \
     --mem=1024G \
     --container-writable \
     --container-image=docker://nishikigi/updated-nemo:25.09-main-v3 \
     --container-mounts=$(pwd):/workdir,/vast:/vast:rw \
     --container-workdir=/workdir \
     bash
```
Then deactivate all conda envs:
```bash
conda deactivate
```
### if you are working on your own cluster, use:
```bash
docker run --rm -it -w /workdir -v $(pwd):/workdir \
  --entrypoint bash \
  --gpus all \
  --ipc=host \
  nishikigi/updated-nemo:25.09-main-v3
```


## 2. Download data (TDC train/valid/test)
### 2.1 Download data from HuggingFace
```bash
python utils/HF_data_download.py
```

## 3. Train the big MoE model with Megatron Bridge
1. Step 1, convert HF model to Megatron Version and save. (**this will finally stop with Error of CUDA OOM, ignore it**)
```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 8 megatron-bridge/bridge-finetune-s1-mini.py \
    --hf_model_save_dir Kiria-Nozan/Intern-S1-Qwen-3-MoE \
    --save_megatron_model
```
2. Step 2, train model with Megatron Bridge.
```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 8 megatron-bridge/bridge-finetune-s1-mini.py
```