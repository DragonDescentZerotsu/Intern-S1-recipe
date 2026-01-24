### To initialize test and train data to fill [TDC_test_prompts_label_scaffold](TDC_test_prompts_label_scaffold) and [TDC_train_prompts_label_scaffold](TDC_train_prompts_label_scaffold). 

Run:
```bash
python DataPrepare/process_tdc_train_test_split.py --target-split train
python DataPrepare/process_tdc_train_test_split.py --target-split test
```
The saved data is already **CoT prompt**.

### Save TDC training prompts and labels by tasks in jsonl format:
```json
{
    "text": "",  # prompt
    "Y": ""     # label
}
```
