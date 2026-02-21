# 1. Run the Slime docker image
## 1.1 On private clusters (no slurm)
Pull docker image
```bash
docker pull slimerl/slime:latest
```
Run docker image
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
  slimerl/slime:latest
```
You can have parallel shells of this running container. To avoid creating files owned by root, start another shell with `root` identity to update the slime repo.
```bash
docker exec -it -u 0:0 <container_id> bash
```
`<container_id>` here is the hostname in `username@hostname` (e.g. `tianang@61b0797d91a1`).

In the new shell with root identity, update to the latest slime repo
```bash
cd /root/slime
git pull
pip install -e . --no-deps --break-system-packages
```
Or we can create a new folder of slime to easier manage code
```bash
cd $PWD/../..
git clone https://github.com/THUDM/slime.git
cd slime
pip install -e . --no-deps --break-system-packages
```
## 1.2 On slurm clusters (Parcc)
1. Get a node shell without pulling container
```bash
srun -p b200-mig45 -N1 -n1 --gpus=1 --time=00:30:00 --pty bash -l
```
2. Pull the docker image
```bash
CACHE=/vast/projects/xia6/apex-gen/tianang/container_cache
mkdir -p $CACHE/{xdg_runtime,enroot_cache,enroot_data,enroot_tmp}
chmod 700 $CACHE/xdg_runtime

export XDG_RUNTIME_DIR=$CACHE/xdg_runtime
export ENROOT_CACHE_PATH=$CACHE/enroot_cache
export ENROOT_DATA_PATH=$CACHE/enroot_data
export ENROOT_TEMP_PATH=$CACHE/enroot_tmp

enroot import -o $CACHE/slime_latest.sqsh docker://slimerl/slime:latest
```
This step saves `slime_latest.sqsh` to `$CACHE/` for repeated use.   

3. Create soft link (软连接) to avoid running out of fiel quota when extracting files of enroot
```bash
mkdir -p ~/.local/share ~/.cache
rm -rf ~/.local/share/enroot ~/.cache/enroot
ln -s $CACHE/enroot_data  ~/.local/share/enroot
ln -s $CACHE/enroot_cache ~/.cache/enroot
```
4. Run the docker image
```bash
CACHE=/vast/projects/xia6/apex-gen/tianang/container_cache
export XDG_RUNTIME_DIR=$CACHE/xdg_runtime
export ENROOT_CACHE_PATH=$CACHE/enroot_cache
export ENROOT_DATA_PATH=$CACHE/enroot_data
export ENROOT_TEMP_PATH=$CACHE/enroot_tmp

P="$(pwd)"

srun --partition=dgx-b200 \
     --pty \
     --nodes=1 \
     --gpus=8 \
     --cpus-per-task=64 \
     --mem=1024G \
     --time=1-00:00:00 \
     --container-writable \
     --container-image=$CACHE/slime_latest.sqsh \
     --container-mounts=$P:$P,$HOME:$HOME,/vast:/vast:rw \
     --container-workdir="$P" \
     --export=ALL,HOME="$HOME" \
     bash -l
```
Multi nodes
```bash
salloc --partition=dgx-b200 -N2 --ntasks-per-node=1 --gpus-per-node=8 \
       --cpus-per-gpu=8 --mem-per-gpu=128G --time=1-00:00:00
```
```bash
CACHE=/vast/projects/xia6/apex-gen/tianang/container_cache
export XDG_RUNTIME_DIR=$CACHE/xdg_runtime
export ENROOT_CACHE_PATH=$CACHE/enroot_cache
export ENROOT_DATA_PATH=$CACHE/enroot_data
export ENROOT_TEMP_PATH=$CACHE/enroot_tmp

P="$(pwd)"

JOBID=4613440  # the salloc job ID
NODE=dgx013  # the node ID you want to use

srun --jobid=$JOBID -N1 -n1 -w $NODE --pty \
     --gpus-per-node=8 --cpus-per-task=64 --mem=0 \
     --container-writable \
     --container-image=$CACHE/slime_latest.sqsh \
     --container-mounts=$P:$P,$HOME:$HOME,/vast:/vast:rw \
     --container-workdir="$P" \
     --export=ALL,HOME="$HOME" \
     bash -l
```
# 2. Update the slime repo in container
### 2.1 Update Slime
```bash
cd $PWD/../..
git clone https://github.com/THUDM/slime.git
cd slime
pip install -e . --no-deps --break-system-packages
``` 
### 2.2 Make a patch for Slime(Megatron-Bridge) to support GLM-4.7-Flash weight transformation
Find file `slime_plugins/mbridge/glm4moe_lite.py`, replace it with:
<details>
<summary>Click to expand code</summary>

```python
from mbridge.core import register_model
from mbridge.models import DeepseekV3Bridge


@register_model("glm4_moe_lite")
class GLM4MoELiteBridge(DeepseekV3Bridge):

    def _build_config(self):
        hf_config = self.hf_config

        # Use getattr to safely access rope_theta with a default value
        # Glm4MoeLiteConfig doesn't expose rope_theta as an attribute
        # even though it exists in config.json
        rope_theta = getattr(hf_config, "rope_theta", 1000000)

        mla_rope_config = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "rope",
        }
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if rope_scaling is not None:
            mla_rope_config.update(rope_scaling)

        moe_layer_freq = [1] * hf_config.num_hidden_layers
        first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", 0)
        for i in range(min(first_k_dense_replace, hf_config.num_hidden_layers)):
            moe_layer_freq[i] = 0

        mtp_args = {}
        num_nextn_predict_layers = getattr(hf_config, "num_nextn_predict_layers", None)
        if num_nextn_predict_layers is not None:
            mtp_args["mtp_num_layers"] = num_nextn_predict_layers
            mtp_args["mtp_loss_scaling_factor"] = 0.1

        base_config = {
            "layernorm_epsilon": hf_config.rms_norm_eps,
            "ffn_hidden_size": hf_config.intermediate_size,
            "qk_layernorm": True,
            # MoE specific
            "moe_ffn_hidden_size": hf_config.moe_intermediate_size,
            "moe_token_dispatcher_type": "alltoall",
            "moe_router_bias_update_rate": 0.001,
            "moe_router_enable_expert_bias": True,
            "moe_router_topk": hf_config.num_experts_per_tok,
            "num_moe_experts": hf_config.n_routed_experts,
            "moe_shared_expert_intermediate_size": hf_config.moe_intermediate_size
            * getattr(hf_config, "n_shared_experts", 1),
            "moe_aux_loss_coeff": getattr(hf_config, "aux_loss_alpha", 0.001),
            "moe_router_load_balancing_type": "none",  # default None for RL
            "moe_shared_expert_overlap": True,
            "moe_grouped_gemm": True,
            "moe_router_score_function": "sigmoid",
            "moe_router_pre_softmax": True,
            "moe_router_topk_scaling_factor": getattr(hf_config, "routed_scaling_factor", 1.0),
            "moe_layer_freq": moe_layer_freq,
            # MLA specific
            "q_lora_rank": hf_config.q_lora_rank,
            "kv_lora_rank": hf_config.kv_lora_rank,
            "qk_head_dim": hf_config.qk_nope_head_dim,
            "qk_pos_emb_head_dim": hf_config.qk_rope_head_dim,
            "v_head_dim": hf_config.v_head_dim,
            "rotary_base": rope_theta,
            "rotary_scaling_factor": mla_rope_config["factor"],
            "rope_type": mla_rope_config["type"],
            "mscale": mla_rope_config["mscale"],
            "mscale_all_dim": mla_rope_config["mscale_all_dim"],
            "beta_fast": mla_rope_config["beta_fast"],
            "beta_slow": mla_rope_config["beta_slow"],
            # mcore 0.12 moe
            "moe_router_dtype": "fp32",
            "disable_bf16_reduced_precision_matmul": True,
            # Other optimizations
            "persist_layer_norm": True,
            "bias_activation_fusion": True,
            "bias_dropout_fusion": True,
        }

        import megatron.core

        megatron_version = getattr(megatron.core, "__version__", "0.0")
        if megatron_version >= "0.14":
            base_config["original_max_position_embeddings"] = mla_rope_config[
                "original_max_position_embeddings"
            ]
        else:
            base_config["max_position_embeddings"] = mla_rope_config[
                "original_max_position_embeddings"
            ]

        base_config.update(mtp_args)
        return self._build_base_config(**base_config)

    def _get_gptmodel_args(self) -> dict:
        """
        Gets the arguments for GPTModel initialization.

        Overrides DeepseekV3Bridge to safely access rope_theta
        via getattr since Glm4MoeLiteConfig may not expose it.
        """
        rope_theta = getattr(self.hf_config, "rope_theta", 1000000)
        return dict(
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=rope_theta,
        )
```
</details>

### 2.3 Update sglang to the latest version to support GLM-4.7-Flash
```bash
cd $PWD/../..
git clone https://github.com/sgl-project/sglang.git
cd sglang/python
pip install -e . --break-system-packages
```
### 2.4 Make a patch for sglang to support transformers v5+
```bash
cd ../..  # in the parent folder of sglang
FILE=sglang/python/sglang/srt/configs/utils.py
cp -a "$FILE" "${FILE}.bak.$(date +%Y%m%d_%H%M%S)"

cat > "$FILE" <<'PY'
from typing import Type

from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    BaseImageProcessor,
    PretrainedConfig,
    ProcessorMixin,
)


def register_image_processor(
    config: Type[PretrainedConfig], image_processor: Type[BaseImageProcessor]
):
    """
    register customized hf image processor while removing hf impl
    """
    # AutoImageProcessor.register(config, None, image_processor, None, exist_ok=True)
    try:
        # transformers v4 时代的签名（有 image_processor_class 这个位置参数）
        AutoImageProcessor.register(config, None, image_processor, None, exist_ok=True)
    except TypeError:
        # transformers v5 时代的新签名
        AutoImageProcessor.register(config, slow_image_processor_class=image_processor, exist_ok=True)



def register_processor(config: Type[PretrainedConfig], processor: Type[ProcessorMixin]):
    """
    register customized hf processor while removing hf impl
    """
    AutoProcessor.register(config, processor, exist_ok=True)
PY

python -m py_compile "$FILE"
```
Actual change in this patch is:
<details>
<summary>Click to see details</summary>

```diff
def register_image_processor(
    config: Type[PretrainedConfig], image_processor: Type[BaseImageProcessor]
):
    """
    register customized hf image processor while removing hf impl
    """
-   AutoImageProcessor.register(config, None, image_processor, None, exist_ok=True)
+   try:
+       # transformers v4 时代的签名（有 image_processor_class 这个位置参数）
+       AutoImageProcessor.register(config, None, image_processor, None, exist_ok=True)
+   except TypeError:
+       # transformers v5 时代的新签名
+       AutoImageProcessor.register(config, slow_image_processor_class=image_processor, exist_ok=True)
```
</details>

### 2.5 Update `transformers`
```bash
pip install -U transformers --break-system-packages
```

### 2.6 Update `flashinfer-jit-cache`
```bash
cd ../..

wget https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.3/flashinfer_jit_cache-0.6.3+cu129-cp39-abi3-manylinux_2_28_x86_64.whl#sha256=b50f1e6d270a95b737606dbc9d48b262fed180100f8143932ff90b87b7d39fa8

pip install flashinfer_jit_cache-0.6.3+cu129-cp39-abi3-manylinux_2_28_x86_64.whl --break-system-packages
```

### 2.7 Install necessary packages for agent tools
```bash
cd projects/Intern-S1  # change to your path
pip install -r requirements_slime.txt --break-system-packages
```

# 3. Transfer checkpoints from HF to Megatron version for Slime training
### 3.1 Download model weight
```bash
hf download zai-org/GLM-4.7-Flash --local-dir checkpoints/megatron/hf_version/GLM-4.7-Flash
```
### 3.2 Transfer the model from HF to Slime
```bash
source your-path-to-slime/scripts/models/glm4.7-30B-A3B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint checkpoints/megatron/hf_version/GLM-4.7-Flash \
    --save checkpoints/megatron/megatron_version/GLM-4.7-Flash
```
# 4. Run the training script
```bash
module load cuda
bash train/RL/Slime/glm4.7-30B-A3B.sh
```

# 5. Reuse and Modification of Fully Async Infrastructure

If you need to write new fully async code for other experiments or new Custom Agents in the future, it is highly recommended to **copy the `generate_tdc_async.py` as your baseline template** rather than the original `slime/examples/fully_async/fully_async_rollout.py`.

### 5.1 Why use `generate_tdc_async.py` instead of the original example?
This implementation contains several critical upgrades and bug fixes compared to the original Slime `fully_async` example:

1. **Fixed Worker Thread & Event Loop Collisions During Evaluation**:
   The original Slime async example will crash (`RuntimeError: Event loop bound`) if `evaluation=True` natively triggers, because `GenerateState` forms a Singleton whose `asyncio.Semaphore` is bound to whichever event loop uses it first. The enhanced version routes the `AsyncRolloutWorker` into the globally shared `slime.utils.async_utils.AsyncLoopThread`, allowing training generation processes and evaluation processes to correctly multiplex without deadlocks or thread boundary crashes.
2. **Fixed Sub-optimal SGLang Queueing**:
   The original code incorrectly breaks out of `while len(active_tasks) < max_concurrent_tasks` on the first iteration, preventing the active tasks buffer from filling up smoothly. The enhanced version correctly populates up to concurrent targets at once.
3. **Robust Timeout and Backpressure Handling**:
   The improved code uses non-blocking puts (`put_nowait`) into the thread queue so that if training is halted (e.g., during Evaluation intervals) and the queue limit of 1000 items is reached, the generator degrades gracefully rather than silently halting the entire training/evaluation loop process in Python.
4. **Deterministic Rollout Returns**:
   Results returned directly from completed futures are re-sorted mathematically (`sorted(completed_groups.keys())`) to feed the Trainer deterministic gradients, avoiding race conditions in gradient calculation caused by dict unordered popping.

### 5.2 Creating a New Async Agent

When starting a new fully async generation, follow these structural rules:

1. **Retain the Global Async Utilities**: 
   Keep `_global_worker`, `AsyncRolloutWorker`, and `generate_rollout_fully_async` exactly as they are without modification. They form the correct Boilerplate.
2. **Implement your own `async def generate(...)`**:
   The logic defining what tokens to send/receive, what Tools to execute locally, and where `responses` or `metadata` arrays are updated should happen inside `generate`.
3. **Point the Shell Script**:
   Both `--rollout-function-path` (for the background generator) and `--custom-generate-function-path` (for the loop content itself) must correctly align with the `generate_rollout_fully_async` and your top-level `generate` functions. Do not mix and match across implementations.