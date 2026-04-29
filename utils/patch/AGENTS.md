# vLLM GPT-OSS SFT Harmony 补丁说明

## 脚本用途

`patch_vllm_gptoss_sft_harmony.py` 用来给当前 Python/conda 环境中安装的 vLLM 打一个兼容补丁。

这个补丁解决的是部分 GPT-OSS SFT 模型在 tool call 场景下输出旧版或不完整 Harmony 片段时，vLLM 的 GPT-OSS Chat 解析器提前报错的问题。补丁会保留原来的严格解析逻辑，只在严格解析失败后，对 legacy SFT 输出做有限的格式修正后再解析。

## 推荐用法

在需要运行 vLLM 的目标 conda 环境里执行：

```bash
conda activate <your_env>
python utils/patch/patch_vllm_gptoss_sft_harmony.py
```

也可以不激活环境，直接使用目标环境的 Python：

```bash
/path/to/conda/envs/<your_env>/bin/python utils/patch/patch_vllm_gptoss_sft_harmony.py
```

脚本会根据“运行它的 Python 环境”自动定位：

```text
vllm/entrypoints/openai/parser/harmony_utils.py
```

因此迁移到别的机器或别的 conda 环境时，不需要修改脚本里的路径。

## 先检查不修改

如果只想确认会修改哪个 vLLM 文件，可以使用：

```bash
python utils/patch/patch_vllm_gptoss_sft_harmony.py --dry-run
```

## 手动指定目标文件

如果 vLLM 是特殊安装方式，自动定位失败，可以手动传入目标文件：

```bash
python utils/patch/patch_vllm_gptoss_sft_harmony.py \
  --target /path/to/site-packages/vllm/entrypoints/openai/parser/harmony_utils.py
```

## 备份和恢复

首次打补丁时，脚本会在目标文件旁边创建备份：

```text
harmony_utils.py.bak-gptoss-sft
```

如果需要恢复原文件：

```bash
python utils/patch/patch_vllm_gptoss_sft_harmony.py --restore
```

如果是手动指定目标文件，恢复时也要传入同一个 `--target`。

## 注意事项

- 请用目标 conda 环境的 Python 运行脚本，否则会给错误环境里的 vLLM 打补丁。
- 脚本会检测是否已经打过补丁；重复运行不会重复修改。
- 如果目标 vLLM 版本中的 `parse_output_into_messages` 函数和脚本预期不一致，脚本会报错并停止。这通常说明 vLLM 版本不同，需要人工确认后再适配补丁。
- 打补丁后需要重启正在运行的 vLLM 服务，已启动的进程不会自动加载修改后的源码。
