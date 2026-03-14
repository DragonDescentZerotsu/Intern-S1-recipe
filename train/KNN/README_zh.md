# TDC 分子任务的 KNN 基线

这个目录放的是基于相似度矩阵的 KNN 基线脚本。它本身不负责生成分子指纹，而是直接消费 `DataPrepare/TDC_mol_fingerprints/` 下面预先算好的产物，用来：

1. 读取 train/valid 的相似度文件，
2. 运行一个简单的加权 KNN 分类器，
3. 在验证集上评估效果，
4. 导出给下游 prompt 构造使用的伪标签。

## 和 `DataPrepare/TDC_mol_fingerprints` 的关系

推荐按下面这条链路理解：

1. `DataPrepare/TDC_mol_fingerprints/compute_fingerprints_and_similarities.py`
   先生成 canonical label map、指纹文件、train/valid 相似度文件。
2. `train/KNN/eval_knn.py`
   读取这些相似度文件，在 `valid` 上做基线评估。
3. `train/KNN/extract_best_knn_labels.py`
   在 `valid` 上选最好的 `k`，再导出 `train` 和 `valid` 的伪标签。
4. `DataPrepare/TDC_prepended/KNN_3/generate_knn_prompts.py`
   再继续读取相似度和伪标签，把 KNN 检索结果拼进 prompt 里。

KNN 的最终相似度来自两种指纹相似度的加权：

```text
weighted_score = 0.8 * Morgan_similarity + 0.2 * Feature_Morgan_similarity
```

对每个 query molecule，脚本会把所有 reference molecule 按这个加权分数排序，取前 `k` 个做多数投票；如果平票，预测为标签 `1`。

## 目录内文件说明

### `eval_knn.py`

只做验证集评估。

功能：

- 读取 `Morgan_similarity` 和 `Feature_Morgan_similarity` 下的 `valid_similarity.pkl`，
- 按 `0.8 / 0.2` 组合两类相似度，
- 在验证集 query 上运行多数投票 KNN，
- 输出 `classification_report` 和 macro-F1。

输入：

- `DataPrepare/TDC_mol_fingerprints/Morgan_similarity/by_task/<task>/valid_similarity.pkl`
- `DataPrepare/TDC_mol_fingerprints/Feature_Morgan_similarity/by_task/<task>/valid_similarity.pkl`

示例：

```bash
python train/KNN/eval_knn.py --tasks BBB_Martins DILI Pgp_Broccatelli -k 3
```

参数：

- `--tasks`：一个或多个任务名。
- `-k`：近邻个数，默认 `3`。

### `extract_best_knn_labels.py`

用于批量提取伪标签。

功能：

- 在验证集上枚举 `k in {3, 6, 9, 12}`，
- 用 macro-F1 选择最优 `k`，
- 用这个最优 `k` 分别对 `train` 和 `valid` 再跑一遍 KNN，
- 把预测结果保存成 `canonical_smiles -> predicted_label` 的 JSON。

输入：

- `DataPrepare/TDC_mol_fingerprints/Morgan_similarity/by_task/<task>/{train,valid}_similarity.pkl`
- `DataPrepare/TDC_mol_fingerprints/Feature_Morgan_similarity/by_task/<task>/{train,valid}_similarity.pkl`

输出：

- `DataPrepare/TDC_mol_fingerprints/KNN_pesudo_labels/by_task/<task>/train_knn_labels.json`
- `DataPrepare/TDC_mol_fingerprints/KNN_pesudo_labels/by_task/<task>/valid_knn_labels.json`

注意：

- 目录名在代码里就是 `KNN_pesudo_labels`，拼写按现有实现保留。
- 当前脚本里的 `tasks` 列表是写死在文件里的，通常需要先改这个列表再运行。

示例：

```bash
python train/KNN/extract_best_knn_labels.py
```

## 推荐使用顺序

如果你是从原始 TDC 任务开始：

```bash
python DataPrepare/TDC_mol_fingerprints/compute_fingerprints_and_similarities.py
python train/KNN/eval_knn.py --tasks Carcinogens_Lagunin -k 3
python train/KNN/extract_best_knn_labels.py
python DataPrepare/TDC_prepended/KNN_3/generate_knn_prompts.py --tasks Carcinogens_Lagunin --splits train valid --top-k 3
```

如果相似度文件已经准备好了，只想跑 KNN：

```bash
python train/KNN/eval_knn.py --tasks Carcinogens_Lagunin -k 3
python train/KNN/extract_best_knn_labels.py
```

## 这些脚本依赖的相似度文件格式

每个相似度 pickle 都是一个字典：

```python
{
    query_smiles: {
        "query_label": 0 or 1,
        "label_0": [(score, ref_smiles), ...],  # 已按分数从高到低排序
        "label_1": [(score, ref_smiles), ...],  # 已按分数从高到低排序
    }
}
```

`eval_knn.py` 和 `extract_best_knn_labels.py` 会把这两个 label-specific 列表重新摊平成一个 reference-score 表，再按加权分数统一排序。

## 实际使用时的注意点

- 这里默认使用的 SMILES key，已经是 `DataPrepare/TDC_mol_fingerprints` 里做过 desalting 和 canonicalization 之后的版本。
- `train_similarity.pkl` 在生成时已经去掉了 self-match，所以 train split 的伪标签不会直接看到自己。
- 如果某个任务的 `valid` 里只有单一类别，脚本就没法计算 macro-F1，只会记录日志说明这个情况。
