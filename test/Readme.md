When using [test_tdc_via_api_AUROC.py](test_tdc_via_api_AUROC.py) to test Intern-s1-mini's tool call performance, host model like this:
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