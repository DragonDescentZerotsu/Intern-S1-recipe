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
## 3. Special functions
### 3.1 Add log request to see the prompt after applying chat template.
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

### 3.2 Alow vLLM to return logprobs
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