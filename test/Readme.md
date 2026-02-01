## When using [test_tdc_via_api_AUROC.py](test_tdc_via_api_AUROC.py) to test Intern-s1-mini's tool call performance, host model like this:
### 1. Intern-s1-mini
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
### 2. GLM-4.7-Flash
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve zai-org/GLM-4.7-Flash \
     --host 0.0.0.0 \
     --port 8000 \
     --tensor-parallel-size 4 \
     --speculative-config.method mtp \
     --speculative-config.num_speculative_tokens 1 \
     --tool-call-parser glm47 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice \
     --served-model-name glm-4.7-flash

```