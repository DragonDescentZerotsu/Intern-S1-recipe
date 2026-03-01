## 1. Install Dependencies

```bash
conda create -n GLM-vllm python=3.12.11
conda activate GLM-vllm
pip install -r requirements.txt
```
for Tinker
```bash
conda create -n tinker python=3.12.11
conda activate tinker
pip install -r requirements_tinker.txt
```

## 2. Prepare Data

### To pre-process and cache the TDC training, validation, and test datasets and populate [DataPrepare/TDC_test_prompts_label_scaffold](DataPrepare/TDC_test_prompts_label_scaffold), [DataPrepare/TDC_valid_prompts_label_scaffold](DataPrepare/TDC_valid_prompts_label_scaffold) and [DataPrepare/TDC_train_prompts_label_scaffold](DataPrepare/TDC_train_prompts_label_scaffold). 

run the following commands:
```bash
python DataPrepare/process_tdc_train_test_split.py --target-split train
python DataPrepare/process_tdc_train_test_split.py --target-split valid
python DataPrepare/process_tdc_train_test_split.py --target-split test
```
The saved data is already **CoT prompt**.

### TDC template prompts from TxGemma are saved in JSONL format, with labels separated by task. Each entry follows the structure below:
```json
{
    "text": "",  # prompt, asking the model to predict the label of the given molecule ((A) or (B))
    "Y": ""     # label, 0 or 1
}
```

To test models/agents on TDC test set, refer to [test_tdc_via_api_F1.py](test/test_tdc_via_api_F1.py)  
To see more agent examples, refer to [agent](agent)

> [!IMPORTANT]
> In terms of label mapping:  
> A -> 0  
> B -> 1

## 3. Test
```bash
python test/test_tdc_via_api_F1.py \
  --task-groups ADME Tox HTS \
  --n-samples 1 \
  --api-base http://localhost:8000/v1 \
  --api-key EMPTY \
  --model "" \
  --num-processes 8 \
  --thinking \
  --enable-tools \
  --log-file \
  --log-file-name "your_log_file.log" \
  --langfuse
```

## 4. Tools for different tasks

There are two kinds of tools:
1. **Baisc Tools**: Tools that are available for all tasks, refer to the `BASIC_TOOLS` variable in [__init__.py](tools/__init__.py)
2. **Task-Specific Tools**: Tool groups that are available for specific tasks, refer to the `TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP` variable in [RDKit_tools.py](tools/RDKit_tools.py)
3. When loading the tools, the function `get_tools_for_task()` in [test/test_tdc_via_api_F1.py](test/test_tdc_via_api_F1.py) will load the tools based on the task groups.

## 5. RL
`GLM-4.7-Flash` RL is supported in [train/RL/Slime](train/RL/Slime), read [train/RL/Slime/Readme.md](train/RL/Slime/Readme.md) for more information.  
`GPT-OSS` RL is supported in [train/RL/Tinker](train/RL/Tinker), read [train/RL/Tinker/Readme.md](train/RL/Tinker/Readme.md) for more information.