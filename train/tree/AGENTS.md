# train/tree

这里放的是树模型基线。当前保留的是 `feature_sources` 这一套结构，因为它更适合后续持续接入新的 feature 来源。

核心原则只有一条：

- 训练逻辑和数据读取逻辑分开
- 训练逻辑和 feature 来源也分开

这样以后你给我新的特征文件时，优先只新增 adapter 或 config，不需要反复改 RF 训练代码。

## 目录约定

- `data.py`
  - 只负责读 `DataPrepare/TDC_no_conflict_labels_salt_removed/{train,valid,test}/*.jsonl`
  - 当前默认字段是 `drug -> smiles`，`Y -> label`

- `feature_sources/`
  - 这里是 feature adapter 层
  - 每种 feature 来源都建议单独一个文件
  - 通过 registry 注册 `source_type`
  - 训练层只拿到一个 `FeatureSource`，不关心底层是 csv、parquet、npy 还是别的

- `preprocessing.py`
  - 负责数值化、缺失值处理、列对齐、可选标准化

- `metrics.py`
  - 放分类指标计算

- `rf_pipeline.py`
  - 放 RF 的共享逻辑
  - 包含：feature config 解析、矩阵准备、调参、最终训练、结果保存
  - 以后如果要接别的树模型，也优先在这里复用公共部分

- `train_random_forest.py`
  - 读取 `best_params.json` 后训练一个最终模型
  - 输出 `train_predictions.csv`、`valid_predictions.csv`、`train_summary.json`、`model_bundle.pkl`
  - 默认会把结果写到 `train/tree/results/<experiment>/<task>/<feature_set>/`

- `train_bundles_from_results.sh`
  - 从一个 tuning experiment 目录里扫描所有 `best_params.json`
  - 逐个调用 `train_random_forest.py` 训练最终模型
  - 把适合下游复用的 bundle 额外收口到 `train/tree/bundles/<experiment>/<task>/<feature_set>/`

- `predict_random_forest_bundle.py`
  - 读取 `model_bundle.pkl` 做预测
  - 支持单条 SMILES、txt 文件、或者带 SMILES 列的 CSV
  - 默认优先从 `train/tree/bundles/<experiment>/<task>/<feature_set>/` 解析 bundle

- `batch_tune_random_forest.py`
  - 负责 RF 调参
  - 支持用 `--tasks` 跑单个或多个任务；不传时默认跑全部任务
  - 严格遵守：只用 `train` 拟合，只在 `valid` 上选超参，或者只在 `train` 上做 CV 选参后再汇报 `valid`
  - 会输出一个汇总表，方便统一查看每个任务的 `valid macro_f1`
  - 默认会把汇总和各任务结果都收口到 `train/tree/results/<experiment>/`

- `configs/`
  - 放 feature source 的 JSON 配置

## 当前已经接好的 feature source

### `fg_top_level_csv`

默认读取：

- `/data1/tianang/Projects/AccFG/FG_feature_extraction/extracted_FG_features/tdc_no_conflict_labels_salt_removed_unique_smiles_top_level_fg_vectors.csv`

规则：

- 第一列 `smiles` 当 key
- 其余列全部作为 feature vector
- 默认会加列名前缀 `fg_top_level__`

默认配置文件：

- `/data1/tianang/Projects/Intern-S1/train/tree/configs/fg_top_level_features.json`

如果训练或调参时不显式传 `--feature-config`，就默认使用这份配置。

### `rdkit_descriptors_and_pka`

默认读取：

- `/data1/tianang/Projects/Intern-S1/DataPrepare/mol_features_for_tree/rdkit_descriptors_and_pka/tdc_no_conflict_labels_salt_removed_unique_smiles_rdkit_descriptors_and_pka.csv`

规则：

- 继续走通用的 `csv_smiles_lookup`
- 第一列 `smiles` 当 key
- 其余列是数值特征
- 默认列名前缀 `rdkit_pka__`

默认配置文件：

- `/data1/tianang/Projects/Intern-S1/train/tree/configs/rdkit_descriptors_and_pka_features.json`

## 怎么新增新的 feature 来源

推荐流程：

1. 在 `feature_sources/` 下面新建一个 adapter 文件
2. 实现一个 `FeatureSource`
3. 最重要的接口保持不变：

```python
def load(self, smiles_list: Sequence[str]) -> pd.DataFrame:
    ...
```

要求：

- 输入是一批按样本顺序排列的 `smiles_list`
- 输出必须和输入行顺序一一对应
- 每一列都是数值 feature
- 不要把 label、task、split 混进 feature frame

4. 调用 `register_feature_source("你的类型名", factory)` 注册
5. 在 `configs/` 下新增 JSON 配置
6. 调参或训练时通过 `--feature-config` 接进去

如果一个实验要拼多个 feature 来源，可以重复传多个 `--feature-config`，训练脚本会按顺序横向拼接。

## 配置格式

### 单个 source

```json
{
  "source_type": "csv_smiles_lookup",
  "name": "my_features",
  "csv_path": "/abs/path/to/features.csv",
  "smiles_column": "smiles",
  "prefix": "my_features__"
}
```

### 多个 source

```json
{
  "feature_set_name": "combo_a",
  "sources": [
    {
      "source_type": "source_a",
      "name": "source_a"
    },
    {
      "source_type": "source_b",
      "name": "source_b"
    }
  ]
}
```

## 运行示例

## 结果目录约定

统一使用：

- `train/tree/results/<experiment>/`
- `train/tree/bundles/<experiment>/`

其中：

- 单任务或批量调参结果：`train/tree/results/<experiment>/<task>/<feature_set>/`
- batch 汇总表：`train/tree/results/<experiment>/all_tasks_tuning_summary.{csv,json}`
- 最终 bundle 导出目录：`train/tree/bundles/<experiment>/<task>/<feature_set>/`

如果你想手动指定实验名，统一传：

```bash
--experiment-name <experiment>
```

单任务调参：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
python train/tree/batch_tune_random_forest.py \
  --tasks BBB_Martins \
  --experiment-name fg_top_level_valid_n40 \
  --n-iter 40 \
  --eval-seeds 0,1,2,3,4 \
  --rf-jobs 1
```

单任务训练最终模型：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
python train/tree/train_random_forest.py \
  --task BBB_Martins \
  --params-json train/tree/results/fg_top_level_valid_n40/BBB_Martins/fg_top_level/best_params.json \
  --seed 0 \
  --rf-jobs 1
```

批量调参所有任务：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
python train/tree/batch_tune_random_forest.py \
  --experiment-name fg_top_level_valid_n40 \
  --n-iter 40 \
  --eval-seeds 0,1,2,3,4 \
  --rf-jobs 1
```

批量导出某个 experiment 的最终 bundle：

```bash
bash train/tree/train_bundles_from_results.sh \
  fg_top_level_plus_rdkit_pka_traincv5_n40_seed0 \
  --rf-jobs 16 \
  --seed 0
```

用 bundle 预测：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
python train/tree/predict_random_forest_bundle.py \
  --experiment-name fg_top_level_plus_rdkit_pka_traincv5_n40_seed0 \
  --task BBB_Martins \
  --smiles "CCO"
```

## 当前默认假设

- 当前任务都是二分类任务
- 当前结果汇报重点是 `valid macro_f1` 和 `valid roc_auc`
- 当前调参策略参考了 `/data1/tianang/Projects/LLM4SD/tune_tdc_rf.py`

如果后面要支持回归任务或多标签任务，建议新增单独入口，而不是把当前二分类 RF baseline 改成一个很难维护的“大一统脚本”。
