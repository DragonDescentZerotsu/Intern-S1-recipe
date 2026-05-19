## ChEMBL 相关脚本

默认在 `node002` 使用 `vllm` conda 环境运行：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python <script>
```

### TDC 分子匹配 ChEMBL

入口：

```bash
DataPrepare/chembl_related/process_scripts/check_tdc_molecules_in_chembl.py
```

功能：检查 `DataPrepare/TDC_no_conflict_labels_salt_removed` 里的分子是否能在 ChEMBL 36 sqlite 中找到对应记录。

常用完整运行：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python DataPrepare/chembl_related/process_scripts/check_tdc_molecules_in_chembl.py --num-workers 64
```

`--num-workers` 控制并行标准化进程数；这台机器核心很多，通常用 32 或 64 即可。

默认输入：

- `DataPrepare/TDC_no_conflict_labels_salt_removed`
- `DataPrepare/chembl_related/chembl_36/chembl_36_sqlite/chembl_36.db`

默认输出：

- `DataPrepare/chembl_related/processed_data/chembl_tdc_overlap`

匹配口径：

- `rdkit`：当前 RDKit 直接从 SMILES 生成 InChIKey。
- `chembl_standardized`：先用 `chembl_structure_pipeline` 标准化，再生成 InChIKey。
- `chembl_parent`：标准化后取 parent/salt-stripped 结构，再生成 InChIKey。

### ChEMBL activity 分布统计

入口：

```bash
DataPrepare/chembl_related/process_scripts/plot_chembl_activity_distribution.py
```

功能：统计 ChEMBL 36 中每个 molecule 的 activity row 数量分布，并画分布图。

默认输入：

- `DataPrepare/chembl_related/chembl_36/chembl_36_sqlite/chembl_36.db`

默认输出：

- `DataPrepare/chembl_related/processed_data/chembl_activity_distribution`

### TDC strict match 的 activity 覆盖率

入口：

```bash
DataPrepare/chembl_related/process_scripts/plot_tdc_chembl_activity_coverage.py
```

功能：统计严格匹配到 ChEMBL 的 TDC 分子中，有多少分子的 activity 数量小于 25，并画图。

默认输出：

- `DataPrepare/chembl_related/processed_data/chembl_tdc_activity_coverage`

### LLM 工具

入口：

```bash
tools/chembl_info.py
```

功能：`chembl_info(smiles)` 返回简洁的 ChEMBL properties；若严格匹配且 activity 数量小于 25，可返回 activity 值和 assay description；否则只返回 properties。无严格匹配时用 RDKit 计算 properties 作为 fallback。

LLM 看到的 tool schema 只有 `smiles` 参数；是否返回 activities 由 Python 侧隐藏参数 `include_activities` 控制。

测试入口：

```bash
test/test_tdc_via_api_F1_TRIM.py
```

常用参数：

- `--tool-mode chembl`：只启用 `chembl_info`。
- `--chembl-include-activities`：默认，返回 properties；若 activity 数量小于 25，也返回 activities。
- `--no-chembl-include-activities`：properties-only 对照实验。
- `--first-turn-tool-choice required`：OpenAI Responses API 第一轮强制调用 tool，适合 smoke test。
- `--openai-reasoning-summary auto`：保存 OpenAI reasoning summary 到 trajectory。

### chembl_info cache

入口：

```bash
DataPrepare/chembl_related/process_scripts/build_chembl_info_cache.py
```

功能：多进程预计算 TDC 分子的 `chembl_info` 返回文本，运行时若 cache 命中则直接返回，避免重复查 mapping、SQLite 和 RDKit fallback。

常用运行：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python DataPrepare/chembl_related/process_scripts/build_chembl_info_cache.py --mode both --num-workers 64
```

可选 `--mode`：

- `both`：同时生成 with-activities 和 properties-only cache。
- `with_activities`：只生成默认实验 cache。
- `properties_only`：只生成 properties-only 对照 cache。

默认输出：

- `DataPrepare/chembl_related/processed_data/chembl_tool_cache/chembl_info_with_activities.jsonl`
- `DataPrepare/chembl_related/processed_data/chembl_tool_cache/chembl_info_properties_only.jsonl`

需要重建 cache 的情况：

- `tdc_unique_molecule_chembl_matches.tsv` 更新。
- `tools/chembl_info.py` 的返回格式、activity 阈值、properties 字段或匹配优先级更新。
- ChEMBL sqlite 数据库版本更新。
