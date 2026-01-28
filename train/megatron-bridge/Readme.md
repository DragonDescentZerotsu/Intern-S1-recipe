# This folder is for Megatron Bridge setup

## 1. Pull my latest modified NeMo docker image
### On slurm cluster, first switch enroot cache path to avoid OOM:
<!-- ```bash
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
``` -->

```bash
export ENROOT_BASE=/vast/projects/xia6/apex-gen/tianang/enroot_cache

export ENROOT_CACHE_PATH=$ENROOT_BASE/cache
export ENROOT_DATA_PATH=$ENROOT_BASE/data
export ENROOT_RUNTIME_PATH=$ENROOT_BASE/runtime
export TMPDIR=$ENROOT_BASE/tmp
export XDG_CACHE_HOME=$ENROOT_BASE/xdg_cache
```
### Then for gpt-oss-20b and other regular models, run:
```bash
P="$(pwd)"

srun --partition=dgx-b200-old-driver \
     --pty \
     --nodes=1 \
     --gpus=8 \
     --cpus-per-task=64 \
     --mem=512G \
     --time=1-00:00:00 \
     --container-writable \
     --container-image=docker://nishikigi/updated-nemo:25.09-main-v3 \
     --container-mounts=$P:$P,$HOME:$HOME,/vast:/vast:rw \
     --container-workdir="$P" \
     --export=ALL,HOME="$HOME" \
     bash -l
```
### For Nemotron 3 Nano, use a specialized docker image
```bash
P="$(pwd)"

srun --partition=dgx-b200-old-driver \
     --pty \
     --nodes=1 \
     --gpus=8 \
     --cpus-per-task=64 \
     --mem=512G \
     --time=1-00:00:00 \
     --container-writable \
     --container-image=docker://nvcr.io/nvidia/nemo:25.11.nemotron_3_nano \
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
### For Nemotron 3 Nano, use a specialized docker image
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
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 8 train/megatron-bridge/bridge-finetune-s1-mini.py \
    --hf_model_save_dir Kiria-Nozan/Intern-S1-Qwen-3-MoE \
    --save_megatron_model
```
2. Step 2, train model with Megatron Bridge.
```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 8 train/megatron-bridge/bridge-finetune-s1-mini.py
```

## 4. Transfer Megatron Bridge checkpoint to HuggingFace Version
For Nemotron 3 Nano, use the nano-v3 docker image (nvcr.io/nvidia/nemo:25.11.nemotron_3_nano)
```bash
python checkpoints/megatron/megatron_to_hf.py export\
     --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16\
     --megatron-path checkpoints/megatron/megatron_version/nemotron-3-30B/TDC_SFT_data_binary_Scaffold_wo_herg-c_ToxCast_butkiewicz/default/checkpoints/iter_0056459\
     --hf-path checkpoints/hub_ready/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```
For gpt-oss-20b, use the updated-nemo docker image (docker://nishikigi/updated-nemo:25.09-main-v3)  
Be careful that the **openai/gpt-oss-20b** model is FP8 and Megatron can't do quantization now, so we need to use the **unsloth/gpt-oss-20b-BF16** as the Huggingface version.
```bash
python checkpoints/megatron/megatron_to_hf.py export\
     --hf-model unsloth/gpt-oss-20b-BF16\
     --megatron-path checkpoints/megatron/megatron_version/gpt-oss-20b/TDC_SFT_data_binary_Scaffold_wo_herg-c_ToxCast_butkiewicz/default/checkpoints/iter_0056459\
     --hf-path checkpoints/hub_ready/gpt-oss-20b
```