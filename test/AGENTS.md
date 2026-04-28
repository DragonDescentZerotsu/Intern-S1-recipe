## TRIM TDC API F1 Eval

`test_tdc_via_api_F1_TRIM.py` is the simplified TRIM-oriented TDC binary-classification evaluator.

Purpose:
- Evaluate API-served LLMs on TDC tasks with macro F1.
- Use salt-removed, label-cleaned data from `DataPrepare/TDC_no_conflict_labels_salt_removed/<split>`.
- Render task prompts through the installed `trim` package, using TRIM prompt templates and each row's `drug` SMILES.
- Optionally expose TRIM's two task-aware tools:
  - `get_mol_properties_and_fg`
  - `compare_similar_mols`

Important defaults:
- Data directory: `DataPrepare/TDC_no_conflict_labels_salt_removed/valid`
- Prompt source: `trim.reasoning.task_user_prompts.render_task_user_message`
- Output format: normal `(A)/(B)` answer parsing only.
- No playbook injection.
- No score-based / 0-100 probability output.
- Default tool mode: `none`
- Default OpenAI reasoning effort: `medium`
- Default max generated tokens: `10240`

Supported API providers:
- `--provider local`: OpenAI-compatible local serve, default `--api-base http://localhost:8001/v1`, default `--api-key EMPTY`.
- `--provider openai`: official OpenAI API, reads `OPENAI_API_KEY` from `.env` or environment when `--api-key` is not provided. OpenAI no-tool runs use Chat Completions; OpenAI tool runs use the Responses API so `reasoning={"effort": ...}` works with function tools.
- `--provider openrouter`: OpenRouter API, default base `https://openrouter.ai/api/v1`; reads `--openrouter-api-key-env`, then common fallback names such as `OPENROUTER_API_KEY_Mark_1`, `OPENROUTER_API_KEY_Mark`, and `OPENROUTER_API_KEY_Haydn`.
- `--provider auto`: infers OpenAI for `gpt-*`/`o*`, OpenRouter for model ids containing `/`, otherwise local.

Tool modes:
- `--tool-mode none`: no tools.
- `--tool-mode properties`: only `get_mol_properties_and_fg`.
- `--tool-mode similar`: only `compare_similar_mols`.
- `--tool-mode both`: expose both TRIM tools.

Compare tool ablations:
- `compare_similar_mols` always runs through the original TRIM runtime first; the evaluator only postprocesses the returned text before sending it back to the model.
- `--similar-tool-feature-view all`: keep the original `compare_similar_mols` text, including neighbor metadata, `properties`, and `functional group differences`.
- `--similar-tool-feature-view properties`: keep neighbor metadata and `properties`, but remove `functional group differences`.
- `--similar-tool-feature-view functional_groups`: keep neighbor metadata and `functional group differences`, but remove `properties`.
- `--similar-tool-feature-view neighbors_only`: keep only neighbor metadata (`label`, `similarity`, `smiles`) and remove both `properties` and `functional group differences`.
- `--similar-tool-property-lines {9,18,27,36}`: optionally keep only the first N property comparison lines per neighbor. This is intended for properties-only ablations; `36` is the current full property block.
- These options only affect `compare_similar_mols`. They do not change `get_mol_properties_and_fg`, TRIM cache payloads, or the underlying TRIM tool implementation.

Useful commands:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python test/test_tdc_via_api_F1_TRIM.py \
  --provider openai \
  --model gpt-5.4-mini \
  --tasks DILI \
  --limit-samples 10 \
  --tool-mode both \
  --num-processes 1
```

Compare tool feature ablation examples:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python test/test_tdc_via_api_F1_TRIM.py \
  --provider openai \
  --model gpt-5.4-mini \
  --tasks BBB_Martins \
  --tool-mode similar \
  --similar-tool-feature-view properties \
  --similar-tool-property-lines 9 \
  --num-processes 1
```

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python test/test_tdc_via_api_F1_TRIM.py \
  --provider openai \
  --model gpt-5.4-mini \
  --tasks BBB_Martins \
  --tool-mode similar \
  --similar-tool-feature-view functional_groups \
  --num-processes 1
```

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python test/test_tdc_via_api_F1_TRIM.py \
  --provider openai \
  --model gpt-5.4-mini \
  --tasks BBB_Martins \
  --tool-mode similar \
  --similar-tool-feature-view neighbors_only \
  --num-processes 1
```

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python test/test_tdc_via_api_F1_TRIM.py \
  --provider local \
  --api-base http://localhost:8001/v1 \
  --api-key EMPTY \
  --model your-served-model-name \
  --tasks DILI \
  --tool-mode properties
```

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python test/test_tdc_via_api_F1_TRIM.py \
  --provider openrouter \
  --model deepseek/deepseek-v3.2 \
  --task-groups Tox \
  --tool-mode none
```

Use `--tasks DILI BBB_Martins` for explicit task selection, or `--task-groups ADME Tox HTS` / `--task-groups all` for grouped runs.

Notes:
- Run this from the repo root.
- On `node002`, use the `vllm` conda environment; TRIM and RDKit dependencies are expected there.
- TRIM tool initialization can be slow because it loads model bundles, feature sources, and cache metadata.
