# When using [test_tdc_via_api_F1.py](test_tdc_via_api_F1.py) to test model's tool call performance, host model like this:
## 1. Intern-s1-mini
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve internlm/Intern-S1-mini \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key EMPTY \
  --dtype auto \
  --served-model-name Intern-S1-mini \
  --max-model-len 32768 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-parser-plugin agent/tool_parser/intern_s1_parser.py \
  --tool-call-parser interns1 \
  --chat-template agent/chat_templates/chat_template_intern-s1_modified.jinja
```
## 2. GLM-4.7-Flash
### vLLM
```bash
CUDA_VISIBLE_DEVICES=4,5 vllm serve zai-org/GLM-4.7-Flash \
     --host 0.0.0.0 \
     --port 8001 \
     --tensor-parallel-size 2 \
     --speculative-config.method mtp \
     --speculative-config.num_speculative_tokens 1 \
     --tool-call-parser glm47 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice \
     --served-model-name glm-4.7-flash
```
### Sglang
```bash
python3 -m sglang.launch_server \
  --model-path checkpoints/megatron/hf_version/GLM-4.7-Flash \
  --tp-size 1 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --mem-fraction-static 0.8 \
  --served-model-name glm-4.7-flash \
  --host 0.0.0.0 \
  --port 8000 \
  --attention-backend triton \
  --speculative-draft-attention-backend triton
```
If not on Blackwell chips, remove attention backend settings
```diff
- --attention-backend triton \
- --speculative-draft-attention-backend triton
```
## 3. gpt-oss
### vLLM
gpt-oss-20b
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve openai/gpt-oss-20b \
     --host 0.0.0.0 \
     --port 8000 \
     --tensor-parallel-size 1 \
     --tool-call-parser openai \
     --enable-auto-tool-choice \
     --served-model-name gpt-oss-20b
```
gpt-oss-120b
```bash
CUDA_VISIBLE_DEVICES=2,3 vllm serve openai/gpt-oss-120b \
     --host 0.0.0.0 \
     --port 8001 \
     --tensor-parallel-size 2 \
     --tool-call-parser openai \
     --enable-auto-tool-choice \
     --served-model-name gpt-oss-120b
```
## 4. Gemma 4
### vLLM
Gemma 4 requires vLLM 0.19.1+ or the `vllm/vllm-openai:gemma4` Docker image; the repo-level `requirements.txt` may be older for existing model runs.

Text-only eval without TRIM tools:
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve google/gemma-4-26B-A4B-it \
     --host 0.0.0.0 \
     --port 8001 \
     --api-key EMPTY \
     --max-model-len 32768 \
     --gpu-memory-utilization 0.90 \
     --served-model-name gemma-4-26B-A4B-it \
     --limit-mm-per-prompt image=0,audio=0
```

With Gemma 4 thinking and tool calling enabled:
`--chat-template` points to the local copy of vLLM's Gemma 4 tool template stored in this repo.
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve google/gemma-4-26B-A4B-it \
  --host 0.0.0.0 \
  --port 8001 \
  --api-key EMPTY \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --served-model-name gemma-4-26B-A4B-it \
  --enable-auto-tool-choice \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --chat-template test/chat_templates/tool_chat_template_gemma4.jinja \
  --language-model-only \
  --async-scheduling
```

Example eval command:
```bash
python test/test_tdc_via_api_F1_TRIM.py \
  --provider local \
  --api-base http://localhost:8001/v1 \
  --api-key EMPTY \
  --model gemma-4-26B-A4B-it \
  --task-groups ADME Tox HTS \
  --tool-mode none \
  --chat-template-kwargs-json '{"enable_thinking": true}' \
  --thinking \
  --num-processes 8 \
  --log-file \
  --log-file-name "gemma4_trim_api_f1_{t_stamp}.log"
```

If the server is launched without `--reasoning-parser gemma4`, remove `--chat-template-kwargs-json '{"enable_thinking": true}'` or expect Gemma thought-channel tags in the raw content.

### HAProxy multi-instance serving
The HAProxy config in [HAProxy_configs/haproxy-vllm.cfg](HAProxy_configs/haproxy-vllm.cfg) load-balances ports 8001-8004 at `http://127.0.0.1:9001`. All vLLM backends must use the same `--served-model-name` because the eval script sends one `model` value to every backend.

Example with one 80GB GPU per Gemma 4 instance:
```bash
GEMMA4_CHAT_TEMPLATE=test/chat_templates/tool_chat_template_gemma4.jinja
CUDA_VISIBLE_DEVICES=0 vllm serve google/gemma-4-26B-A4B-it --host 127.0.0.1 --port 8001 --api-key EMPTY --max-model-len 16384 --gpu-memory-utilization 0.90 --served-model-name gemma-4-26B-A4B-it --enable-auto-tool-choice --reasoning-parser gemma4 --tool-call-parser gemma4 --chat-template "$GEMMA4_CHAT_TEMPLATE" --limit-mm-per-prompt image=0,audio=0 --async-scheduling
CUDA_VISIBLE_DEVICES=1 vllm serve google/gemma-4-26B-A4B-it --host 127.0.0.1 --port 8002 --api-key EMPTY --max-model-len 16384 --gpu-memory-utilization 0.90 --served-model-name gemma-4-26B-A4B-it --enable-auto-tool-choice --reasoning-parser gemma4 --tool-call-parser gemma4 --chat-template "$GEMMA4_CHAT_TEMPLATE" --limit-mm-per-prompt image=0,audio=0 --async-scheduling
CUDA_VISIBLE_DEVICES=2 vllm serve google/gemma-4-26B-A4B-it --host 127.0.0.1 --port 8003 --api-key EMPTY --max-model-len 16384 --gpu-memory-utilization 0.90 --served-model-name gemma-4-26B-A4B-it --enable-auto-tool-choice --reasoning-parser gemma4 --tool-call-parser gemma4 --chat-template "$GEMMA4_CHAT_TEMPLATE" --limit-mm-per-prompt image=0,audio=0 --async-scheduling
CUDA_VISIBLE_DEVICES=3 vllm serve google/gemma-4-26B-A4B-it --host 127.0.0.1 --port 8004 --api-key EMPTY --max-model-len 16384 --gpu-memory-utilization 0.90 --served-model-name gemma-4-26B-A4B-it --enable-auto-tool-choice --reasoning-parser gemma4 --tool-call-parser gemma4 --chat-template "$GEMMA4_CHAT_TEMPLATE" --limit-mm-per-prompt image=0,audio=0 --async-scheduling
```

Start HAProxy:
```bash
haproxy -f test/HAProxy_configs/haproxy-vllm.cfg
```

Then point eval at HAProxy:
```bash
python test/test_tdc_via_api_F1_TRIM.py \
  --provider local \
  --api-base http://127.0.0.1:9001/v1 \
  --api-key EMPTY \
  --model gemma-4-26B-A4B-it \
  --task-groups ADME Tox HTS \
  --tool-mode none \
  --chat-template-kwargs-json '{"enable_thinking": true}' \
  --thinking \
  --num-processes 32
```

Formal Gemma 4 TRIM-tools F1 run through HAProxy:
```bash
/home/tianang/anaconda3/bin/conda run -n reasonv python test/test_tdc_via_api_F1_TRIM.py \
  --provider local \
  --api-base http://127.0.0.1:9001/v1 \
  --api-key EMPTY \
  --model gemma-4-26B-A4B-it \
  --task-groups ADME Tox HTS \
  --tool-mode both \
  --first-turn-tool-choice auto \
  --chat-template-kwargs-json '{"enable_thinking": true}' \
  --thinking \
  --n-samples 1 \
  --num-processes 32 \
  --max-retry 4 \
  --log-file \
  --log-file-name "gemma4_trim_tools_api_f1_{t_stamp}.log"
```

For a quick tool round-trip smoke test, add `--first-turn-tool-choice required` and a small `--limit-samples`, but keep `auto` for the formal eval above.

## 5. Special functions
### 5.1 Add log request to see the prompt after applying chat template.
```bash
export VLLM_LOGGING_LEVEL=DEBUG
CUDA_VISIBLE_DEVICES=4,5 vllm serve zai-org/GLM-4.7-Flash \
     --host 0.0.0.0 \
     --port 8001 \
     --tensor-parallel-size 2 \
     --speculative-config.method mtp \
     --speculative-config.num_speculative_tokens 1 \
     --tool-call-parser glm47 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice \
     --served-model-name glm-4.7-flash \
     --enable-log-requests \
     --max-log-len 20000 \
     --enable-log-outputs
```
To turn off vllm debuging mode:
```bash
export VLLM_LOGGING_LEVEL=INFO
```

### 5.2 Alow vLLM to return logprobs
For [test_tdc_via_api_F1_optimal_threshold.py](test_tdc_via_api_F1_optimal_threshold.py), add `--max-logprobs` to the vLLM host command.
```bash
CUDA_VISIBLE_DEVICES=4,5 vllm serve zai-org/GLM-4.7-Flash \
     --host 0.0.0.0 \
     --port 8001 \
     --tensor-parallel-size 2 \
     --speculative-config.method mtp \
     --speculative-config.num_speculative_tokens 1 \
     --tool-call-parser glm47 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice \
     --served-model-name glm-4.7-flash \
     --max-logprobs 120
```
Change `--max-logprobs 20` to `--max-logprobs 100` to get more logprobs.
