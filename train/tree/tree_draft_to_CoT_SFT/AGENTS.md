# tree_draft_to_CoT_SFT 工作约定

这个目录负责把 `train/tree/tree_reasoning_processes/...` 里的树路径草稿，继续整理成适合：

- DeepResearch 查 literature threshold / heuristic
- 大模型改写成高质量 reasoning CoT
- 最终沉淀为 SFT 训练语料

如果这条主线的计划发生明显变化，或者新增了关键脚本 / 输出文件 / prompt 模版，之后继续合作时要同步更新这个文件。

## 当前主线流程

### 第 1 步：确认 task 的稳定 feature 空间

优先来源：

- `train/tree/results/<experiment>/<task>/<feature_set>/best_params.json`
- `train/tree/results/<experiment>/<task>/<feature_set>/train_summary.json`

关键字段：

- `surviving_feature_names`

原因：

- 原始 feature CSV 只告诉我们“理论上有哪些列”。
- 真正进入该 task 训练流程的稳定特征，还要经过 [train/tree/preprocessing.py](/data1/tianang/Projects/Intern-S1/train/tree/preprocessing.py) 的 NaN 列清理。
- 这里的逻辑是：训练集里某一列只要出现任意 NaN，这整列就会被丢掉。
- 因此，对某个 task 来说，最稳定、最贴近真实训练输入的 feature 清单，不是单看原始 CSV header，而是看该 task 结果文件里的 `surviving_feature_names`。

结论：

- 生成 playbook prompt 时，优先把 `surviving_feature_names` 当作 task-specific 的稳定候选 feature 空间。
- reasoning JSON 里的 feature 统计只作为优先级和覆盖情况参考，不再作为唯一的 feature discovery 来源。

### 第 2 步：再生成 / 使用 reasoning JSON

上游入口：

- `train/tree/select_reasoning_trees.py`
- `train/tree/batch_select_reasoning_trees.py`

它会生成：

- `train/tree/tree_reasoning_processes/<experiment>/<task>/<feature_set>/*.json`

这些 JSON 是后续一切整理脚本的输入来源。

注意：

- 这里不要求“导出森林里所有 tree 的所有路径”。
- 但如果想让 threshold playbook 更贴近后续 CoT 改写时真正会遇到的 feature 优先级，仍然建议先为某个 task 生成一批 reasoning JSON。
- 如果要为整个 task 批量生成 trace，优先使用 `train/tree/batch_select_reasoning_trees.py`，不要手写 shell loop。
- 对最终 SFT 训练语料主线，优先生成 `train` split 的 trace；`valid` 更适合调试、抽样检查和 prompt 迭代。
- 现在 `prepare_task_threshold_research_brief.py` 会把 reasoning JSON 用在：
  - 识别哪些 surviving features 已经在草稿里真正出现过
  - 统计哪些 feature 在 `important_feature_set` 里更重要
  - 统计哪些 feature 在 `reasoning_steps` 里更常被路径使用
- 因此，reasoning JSON 覆盖越多，后面的 feature 排序和 research priority 越稳。
- 但即使 reasoning JSON 还不全，playbook 的候选 feature 空间也不会只靠 observed paths 决定。

### 第 3 步：整理成 task-specific research brief

当前脚本：

- `train/tree/tree_draft_to_CoT_SFT/prepare_task_threshold_research_brief.py`

用途：

- 读取某个 task 的 result artifacts 里的 `surviving_feature_names`
- 结合 reasoning JSON 的 observed 统计
- 自动抽取非 functional-group 的特征
- 把特征拆成两层：
  - `threshold_research_features`
  - `qualitative_rewrite_features`
- 生成给 DeepResearch 用的 prompt-ready 输出

默认策略：

- 默认 `--feature-universe-source hybrid`
- 只看 `valid` split
- 默认只包含 `rdkit,pka`
- 默认排除 `fg_top_level`

原因：

- `fg_top_level` 这类 functional-group indicator/count 特征，当前不作为 literature threshold research 的对象
- 它们更适合留到后续 CoT 改写阶段当作定性结构证据

当前 `feature-universe-source` 语义：

- `hybrid`
  - 用 task 结果文件里的 `surviving_feature_names` 作为 playbook 候选主集合
  - 用 reasoning JSON 统计做优先级参考
- `results_surviving`
  - 只基于 `surviving_feature_names`
- `reasoning_observed`
  - 只基于已经观察到的 reasoning JSON feature
  - 只适合探索，不是当前推荐默认值

### 第 4 步：DeepResearch 产出 threshold playbook

DeepResearch prompt 模版：

- `train/tree/tree_draft_to_CoT_SFT/prompt_templates/deepresearch_threshold_playbook_prompt_template.md`

脚本渲染后的 task-specific prompt：

- `train/tree/tree_draft_to_CoT_SFT/outputs/<experiment>/<task>/<feature_set>/deepresearch_threshold_playbook_prompt_filled.md`

这里 DeepResearch 只应该重点覆盖：

- `threshold_research_features`

而不是所有出现过的特征。

### 第 5 步：大模型改写草稿为高质量 CoT

CoT 重写 prompt 模版：

- `train/tree/tree_draft_to_CoT_SFT/prompt_templates/rewrite_tree_draft_to_cot_prompt_template.md`

这个阶段要结合：

- `path_level_reasoning_note`
- `reasoning_steps`
- `important_feature_set`
- DeepResearch 产出的 threshold playbook
- `qualitative_rewrite_features`

目标：

- 让最终 CoT 看起来像模型在自主分析分子
- 不暴露树结构术语
- 只在安全时把模型阈值替换成 literature threshold
- 尽量把不同 sample / 不同 tree 中不一致的原始 learned threshold 归一到更稳定的 literature-facing 锚点
- 没有稳定阈值的 feature 只作为定性证据使用
- functional-group 相关步骤尽量改写成存在 / 不存在 / 大致一个 / 多个这类自然语言判断，而不是保留 `0.5`、`1.5` 这类整数计数切分痕迹

当前重写 prompt 额外强调：

- threshold playbook 的作用不只是“可选替换阈值”，而是帮助不同树、不同样本之间形成更统一的方法论
- 对没有稳定文献阈值的 property，要优先写成趋势、倾向、机制线索，而不是保留任意 learned threshold
- 每一步都应带有一定分析感，让最终 CoT 更像真实思考，而不是 symbolic execution

## 双层特征输出的含义

### `threshold_research_features`

这些特征更适合送去查 literature threshold / common cutoff / heuristic range。

典型包括：

- `logD`
- `logP`
- `TPSA`
- `MW`
- `HBD/HBA`
- `rotatable bonds`
- `ionizable-site counts`
- 一些常见 ring / heteroatom / count 类描述符

### `qualitative_rewrite_features`

这些特征不一定适合做 literature threshold research，但在 CoT 重写时仍然有价值。

典型包括：

- 细粒度 `fr_...` fragment count
- warning / boolean state feature
- 一些更偏技术性的 descriptor

用法：

- 可以进入最终 CoT
- 但不要硬写成文献阈值规则
- 更适合写成：
  - 某种结构线索
  - 某种方向性支持
  - 某种定性化学证据

## 当前关键文件

### 脚本

- `train/tree/tree_draft_to_CoT_SFT/prepare_task_threshold_research_brief.py`
- `train/tree/tree_draft_to_CoT_SFT/render_rewrite_tree_draft_prompt.py`

### Prompt 模版

- `train/tree/tree_draft_to_CoT_SFT/prompt_templates/deepresearch_threshold_playbook_prompt_template.md`
- `train/tree/tree_draft_to_CoT_SFT/prompt_templates/rewrite_tree_draft_to_cot_prompt_template.md`

### Playbook 目录

- `playbooks/tree_thresholds/<task>.md`

当前约定：

- rewrite prompt 填充脚本默认从这个目录读取 task 对应的 threshold playbook
- 文件名按 task 命名，例如 `playbooks/tree_thresholds/BBB_Martins.md`

### 输出目录

- `train/tree/tree_draft_to_CoT_SFT/outputs/<experiment>/<task>/<feature_set>/`

当前会生成：

- `task_threshold_research_brief.json`
- `task_threshold_research_brief.md`
- `deepresearch_threshold_playbook_prompt_filled.md`
- `rewrite_prompts/<sample_tag>__tree<rank>_rewrite_prompt_filled.md`

## 当前 rewrite prompt 填充方式

当前入口：

- `train/tree/tree_draft_to_CoT_SFT/render_rewrite_tree_draft_prompt.py`

默认输入：

- 一条 reasoning JSON
- 该 task 在 `playbooks/tree_thresholds/<task>.md` 下的 playbook
- 指定 `tree_rank` 对应的 `path_level_reasoning_note`

当前默认不再把下面这些内容直接塞进 rewrite prompt，以避免 prompt 过长：

- `reasoning_steps`
- `important_feature_set`

推荐用法：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
python train/tree/tree_draft_to_CoT_SFT/render_rewrite_tree_draft_prompt.py \
  --reasoning-json train/tree/tree_reasoning_processes/fg_top_level_plus_rdkit_pka_easy_to_NLP_Lv1_traincv5_n40_eval0_jobs16/BBB_Martins/fg_top_level+rdkit_descriptors_and_pka_easy_to_NLP_Lv1/train_sample_0__top30__trees5.json \
  --tree-rank 1
```

或者用目录坐标：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm \
python train/tree/tree_draft_to_CoT_SFT/render_rewrite_tree_draft_prompt.py \
  --experiment-name fg_top_level_plus_rdkit_pka_easy_to_NLP_Lv1_traincv5_n40_eval0_jobs16 \
  --task BBB_Martins \
  --feature-set 'fg_top_level+rdkit_descriptors_and_pka_easy_to_NLP_Lv1' \
  --sample-tag train_sample_0__top30__trees5 \
  --tree-rank 1
```

## 当前已知约束

- DeepResearch prompt 当前只吃 `threshold_research_features`
- 在 `hybrid` / `results_surviving` 模式下，真正进入 playbook 的 feature 必须来自 task 结果文件里的 `surviving_feature_names`
- `qualitative_rewrite_features` 当前保存在 summary JSON / markdown 里，供后续 CoT 重写阶段使用
- functional-group (`fg_top_level`) 特征默认不进入 literature threshold research
- 如果后面要把 `qualitative_rewrite_features` 再进一步分层，例如：
  - strongly useful qualitative
  - low-priority qualitative
  - omit by default
  需要更新这个文件
