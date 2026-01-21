### 1. vLLM serving Intern-S1-mini (1 x A100):
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

### 2. vLLM serving Intern-S1 (8 x A100):
```bash
vllm serve internlm/Intern-S1 \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key EMPTY \
  --dtype auto \
  --served-model-name Intern-s1 \
  --max-model-len 32768 \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-parser-plugin agent/tool_parser/intern_s1_parser.py \
  --tool-call-parser interns1 \
  --chat-template agent/chat_templates/chat_template_intern-s1_modified.jinja
```

### 3. vLLM serving DeepSeek V3.2 (on Parcc, 4 x GB200):
注意可能要安装DeepGEMM (maybe you have to install DeepGEMM)
```bash
# 先把CUDA拉回来 (load CUDA)
module avail cuda
module load cuda
# 然后安装 (install DeepGEMM)
pip install -U "git+https://github.com/deepseek-ai/DeepGEMM@main" --no-build-isolation
```
还要注意如果使用 Antigravity 的话进去的 shell 可能是 non-login shell，所以会看不到 slurm 的任何命令，要先执行 (also note that if you use Antigravity, the shell may be non-login shell, so you can't see any slurm commands, you have to execute)：
```bash
# 变成 login shell (change to login shell)
exec bash -l
```
Then serve DeepSeek V3.2 as:
```bash
vllm serve deepseek-ai/DeepSeek-V3.2 \
  --tensor-parallel-size 4 \
  --tokenizer-mode deepseek_v32 \
  --tool-call-parser deepseek_v32 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v3 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 51200 \
  --max-num-seqs 128
```
