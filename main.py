import json
from huggingface_hub import hf_hub_download
from tdc.single_pred import ADME

# 下载 TDC 提示模板
tdc_prompts_filepath = hf_hub_download(
    repo_id="google/txgemma-9b-predict",
    filename="tdc_prompts.json",
)
tdc_prompts_json = json.load(open(tdc_prompts_filepath))

# 选择一个任务和输入
task_name = "BBB_Martins"
input_type = "{Drug SMILES}"
drug_smiles = "CN1C(=O)CN=C(C2=CCCCC2)c2cc(Cl)ccc21"

# 用实际输入替换占位符
prompt = tdc_prompts_json[task_name].replace(input_type, drug_smiles)
print(prompt)

data = ADME(name='BBB_Martins')
split = data.get_split()
test_df = split['test']

print(test_df)