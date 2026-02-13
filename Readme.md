## 1. Install Dependencies

```bash
conda create -n GLM-vllm python=3.12.11
conda activate GLM-vllm
pip install -r requirements.txt
```
for tinker
```bash
conda create -n tinker python=3.12.11
conda activate tinker
pip install -r requirements_tinker.txt
```

## 2. Prepare Data

### To pre-process and cache the TDC training and test datasets and populate [DataPrepare/TDC_test_prompts_label_scaffold](DataPrepare/TDC_test_prompts_label_scaffold) and [DataPrepare/TDC_train_prompts_label_scaffold](DataPrepare/TDC_train_prompts_label_scaffold). 

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
    "text": "",  # prompt
    "Y": ""     # label
}
```

To test models/agents on TDC test set, refer to [test_tdc_via_api.py](test_tdc_via_api.py)  
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
  --log-file-name "Tools_Intern-S1-mini-distill_DeepSeek_V32_1_epoch_{t_stamp}_hERG_Karim.log" \
  --langfuse
```