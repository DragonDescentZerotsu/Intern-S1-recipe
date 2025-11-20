# This repo is for Intern-S1 recipe experiment

## 1. pull latest modified NeMo docker image
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
if you are working on your own cluster, use:
```bash
docker run --rm -it -w /workdir -v $(pwd):/workdir \
  --entrypoint bash \
  --gpus all \
  --ipc=host \
  nishikigi/updated-nemo:25.09-main-v3
```


## 2. Download data
```bash
python utils/HF_data_download.py
```

## 3. Train the big MoE model with Megatron Bridge
1. Step 1, save HF model to Megatron Version. (**this will finally stop with Error of CUDA OOM, ignore it**)
```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 8 megatron-bridge/bridge-finetune-s1-mini.py \
    --hf_model_save_dir Kiria-Nozan/Intern-S1-Qwen-3-MoE \
    --save_megatron_model
```
2. Step 2, train model with Megatron Bridge.
```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 8 megatron-bridge/bridge-finetune-s1-mini.py
```
-----
## Current Issues:
When training the Megatron model, you will will see the training process stuck at:
>Training ... \
Setting rerun_state_machine.current_iteration to 0...

<pre>
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA B200                    On  |   00000000:1B:00.0 Off |                    0 |
| N/A   29C    P0            187W / 1000W |  100536MiB / 183359MiB |    100%      Default | 
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA B200                    On  |   00000000:43:00.0 Off |                    0 |
| N/A   31C    P0            197W / 1000W |   98232MiB / 183359MiB |    100%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA B200                    On  |   00000000:52:00.0 Off |                    0 |
| N/A   34C    P0            194W / 1000W |   98232MiB / 183359MiB |    100%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA B200                    On  |   00000000:61:00.0 Off |                    0 |
| N/A   33C    P0            194W / 1000W |   98232MiB / 183359MiB |    100%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA B200                    On  |   00000000:9D:00.0 Off |                    0 |
| N/A   31C    P0            187W / 1000W |   94500MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA B200                    On  |   00000000:C3:00.0 Off |                    0 |
| N/A   30C    P0            190W / 1000W |   94500MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA B200                    On  |   00000000:D1:00.0 Off |                    0 |
| N/A   34C    P0            193W / 1000W |   94500MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA B200                    On  |   00000000:DF:00.0 Off |                    0 |
| N/A   34C    P0            190W / 1000W |   94500MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
 \
+-----------------------------------------------------------------------------------------+ \
| Processes:                                                                              | \
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory | \
|        ID   ID                                                               Usage      | \
|=========================================================================================| \
|    0   N/A  N/A         2709700      C   /opt/venv/bin/python                  10052... | \
|    1   N/A  N/A         2709701      C   /opt/venv/bin/python                  98218MiB | \
|    2   N/A  N/A         2709702      C   /opt/venv/bin/python                  98218MiB | \
|    3   N/A  N/A         2709703      C   /opt/venv/bin/python                  98218MiB | \
|    4   N/A  N/A         2709704      C   /opt/venv/bin/python                  94486MiB | \
|    5   N/A  N/A         2709705      C   /opt/venv/bin/python                  94486MiB | \
|    6   N/A  N/A         2709706      C   /opt/venv/bin/python                  94486MiB | \
|    7   N/A  N/A         2709707      C   /opt/venv/bin/python                  94486MiB | \
+-----------------------------------------------------------------------------------------+ \
</pre>
