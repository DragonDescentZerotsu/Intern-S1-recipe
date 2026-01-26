# This folder is for Megatron Bridge setup

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
```bash
P="$(pwd)"

srun --partition=dgx-b200-old-driver \
     --pty \
     --nodes=1 \
     --gpus=8 \
     --cpus-per-task=64 \
     --mem=512G \
     --container-writable \
     --container-image=docker://nishikigi/updated-nemo:25.09-main-v3 \
     --container-mounts=$P:$P,$HOME:$HOME,/vast:/vast:rw \
     --container-workdir="$P" \
     --export=ALL,HOME="$HOME" \
     bash -l
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
### To share more content between docker container and the host:
```bash
P="$(pwd)"
docker run --rm -it \
  -u "$(id -u)":"$(id -g)" \
  -v /data2/tianang:/data2/tianang \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v "$HOME":"$HOME" \
  -v "$P":"$P" \
  -e HOME="$HOME" \
  -w "$P" \
  --entrypoint bash \
  --gpus all \
  --ipc=host \
  nishikigi/updated-nemo:25.09-main-v3
```
### For Nemotron 3 Nano
```bash
P="$(pwd)"
docker run --rm -it \
  -u "$(id -u)":"$(id -g)" \
  -v /data2/tianang:/data2/tianang \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v "$HOME":"$HOME" \
  -v "$P":"$P" \
  -e HOME="$HOME" \
  -w "$P" \
  --entrypoint bash \
  --gpus all \
  --ipc=host \
  nvcr.io/nvidia/nemo:25.11.nemotron_3_nano
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