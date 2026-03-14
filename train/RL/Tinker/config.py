from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

from dotenv import load_dotenv
# load TINKER_API_KEY from .env file
load_dotenv(PROJECT_ROOT / ".env")

# ======================================================================
# TASK GROUPS & CONFIG
# ======================================================================

TASK_GROUPS = {
    "ADME": ["PAMPA_NCATS","HIA_Hou","BBB_Martins","Pgp_Broccatelli","Bioavailability_Ma",
             "CYP2C9_Substrate_CarbonMangels","CYP2D6_Substrate_CarbonMangels",
             "CYP3A4_Substrate_CarbonMangels","CYP1A2_Veith","CYP2C19_Veith",
             "CYP2C9_Veith","CYP2D6_Veith","CYP3A4_Veith"],
    "Tox": ["hERG_Karim","Carcinogens_Lagunin","Skin_Reaction","hERG","DILI","ClinTox","AMES"],
    "HTS": ["HIV","SARSCoV2_3CLPro_Diamond","SARSCoV2_Vitro_Touret"],
}
TASK_TO_GROUP = {t: g for g, ts in TASK_GROUPS.items() for t in ts}

# Tinker model string -> HuggingFace tokenizer name
TINKER_TO_HF = {
    "GPT-OSS-120B": "openai/gpt-oss-120b",
    "GPT-OSS-20B": "openai/gpt-oss-20b",
    "openai/gpt-oss-20b": "openai/gpt-oss-20b"
    # Qwen models: Tinker string == HF string, no mapping needed
}

EXCLUDED_TASKS = (
        'hERG_Karim',  # 2690
        'Carcinogens_Lagunin',  # 56
        'Skin_Reaction',  # 82
        'hERG',  # 132
        'DILI',  # 96
        'ClinTox',  # 297
        'AMES',  # 1457
        'Tox21',  # 15584
        # -----------------------------------------
        'herg_central_hERG_inhib',  # 61379    leave out
        'ToxCast',  # 307282    leave out
        'PAMPA_NCATS',  # 408
        'HIA_Hou',  # 117
        # 'BBB_Martins',  # 406
        'Pgp_Broccatelli',  # 245
        'Bioavailability_Ma',  # 128
        'CYP2C9_Substrate_CarbonMangels',  # 135
        'CYP2D6_Substrate_CarbonMangels',  # 135
        'CYP3A4_Substrate_CarbonMangels',  # 135
        'CYP1A2_Veith',  # 2517
        'CYP2C19_Veith',  # 2534
        'CYP2C9_Veith',  # 2419
        'CYP2D6_Veith',  # 2626
        'CYP3A4_Veith',  # 2467
        'HIV',  # 8225
        'SARSCoV2_3CLPro_Diamond',  # 176
        'SARSCoV2_Vitro_Touret',  # 298
        # -----------------------------------------
        'butkiewicz',  # 401997    leave out
        'SAbDab_Chen',
        'HuRI',
        'Weber',
        'phase1',
        'phase2',
        'phase3',
        'MHC1_IEDB-IMGT_Nielsen',  # 37197
        'MHC2_IEDB_Jensen'  # 26856
    )

save_every_dict = {
    "BBB_Martins": 44,
    "DILI": 5,
}

eval_every_dict = {
    "BBB_Martins": 2,
    "DILI": 1,
}

@dataclass
class TrainConfig:
    model_name: str = "openai/gpt-oss-20b"
    lora_rank: int = 32
    learning_rate: float = 1e-5
    lr_schedule: str = "cosine_decay"  # "none" or "cosine_decay"
    min_learning_rate_ratio: float = 0.1  # final lr = learning_rate * ratio for cosine decay
    batch_size: int = 32
    group_size: int = 8
    target_effective_groups: Optional[int] = None  # defaults to current batch size
    max_group_rollout_rounds: int = 4  # max rollout rounds per sample to recover a non-zero-adv group
    max_tokens: int = 16384
    max_turns: int = 40
    reward_format_bonus: float = 0
    reward_use_tools: float = 0.01
    reward_length_penalty: bool = False
    checkpoint_strategy: str = "best_eval"  # "best_eval" or "interval"
    save_final_state: bool = False
    save_every: int = save_every_dict['BBB_Martins']
    eval_every: int = eval_every_dict['BBB_Martins']
    eval_max_samples: int = 1000
    eval_temperature: float = 1
    eval_n_samples: int = 1
    eval_max_retries: int = 4
    use_tools: bool = False
    filter_easy_samples: bool = True  # whether to downsample zero-adv easy samples in later epochs
    easy_sample_retry_prob: float = 0.3  # chance to retry a previously easy sample in the next epoch
    wandb_project: str = "tdc-tinker-grpo"
    wandb_run_id: Optional[str] = None #"hbnj1kto"
    wandb_name: Optional[str] = "GRPO BBB 2 epochs keep easy samples & KNN_3 prepend"
    data_dir: Path = PROJECT_ROOT / "DataPrepare/TDC_prepended/KNN_3/train"
    log_path: str = PROJECT_ROOT / "logs/tinker/BBB/2-epochs-grpo-keep-easy-samples-knn-prepend"
    task: Optional[str] = 'BBB_Martins' # None
    exclude_tasks: tuple = EXCLUDED_TASKS
    resume_from: Optional[str] = None # "tinker://be372b79-3da0-589d-b696-ebb65c3a86ec:train:0/weights/step_000470"
    resume_step: int = 470
    data_seed: int = 1
    rollout_save_dir: Optional[str] = PROJECT_ROOT / "train/RL/Tinker/rollouts/BBB/2-epochs-grpo-keep-easy-samples-knn-prepend"
    epochs: int = 2
    playbook_dir: Optional[str] = None #PROJECT_ROOT / "playbooks/discover_deepresearch"  # None

@dataclass
class EvalConfig:
    test_data_dir: Path = PROJECT_ROOT / "DataPrepare/TDC_prepended/KNN_3/valid"
    eval_tasks: Optional[list[str]] = None
    skip_tasks: tuple = EXCLUDED_TASKS
    max_tokens: int = 16384
    max_turns: int = 40
    temperature: float = 0.6
    top_p: float = 0.95
    n_samples: int = 1
    use_tools: bool = False
    max_samples_per_task: Optional[int] = None
    log_dir: Optional[str] = None
    verbose: bool = True
    eval_max_retries: int = 4
    playbook_dir: Optional[str] = None # PROJECT_ROOT / "playbooks/discover_deepresearch"  # None

cfg = TrainConfig()

# Global Constants
MAX_CONTEXT_TOKENS = 120_000
SAMPLE_TIMEOUT_SEC = 900
STOP_TOKEN_IDS = [200012, 200002]  # <|call|>, <|return|>
