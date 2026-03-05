from huggingface_hub import HfApi
import os

api = HfApi()

repo_id="Kiria-Nozan/TDC_train_prompts_label_sm_wo_herg-c_ToxCast_butkiewicz"

api.create_repo(repo_id=repo_id, repo_type="dataset")

api.upload_folder(
    folder_path="DataPrepare/TDC_train_prompts_label_sm_wo_herg-c_ToxCast_butkiewicz",
    repo_id=repo_id,  # 例如 "Kiria-Nozan/TDC-train-prompts-label-scaffold"
    repo_type="dataset",
)