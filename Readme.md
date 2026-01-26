## 1. Prepare Data

### To pre-process and cache the TDC training and test datasets and populate [DataPrepare/TDC_test_prompts_label_scaffold](DataPrepare/TDC_test_prompts_label_scaffold) and [DataPrepare/TDC_train_prompts_label_scaffold](DataPrepare/TDC_train_prompts_label_scaffold). 

run the following commands:
```bash
python DataPrepare/process_tdc_train_test_split.py --target-split train
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