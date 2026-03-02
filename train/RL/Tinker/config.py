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
        # 'DILI',  # 96
        'ClinTox',  # 297
        'AMES',  # 1457
        'Tox21',  # 15584
        # -----------------------------------------
        'herg_central_hERG_inhib',  # 61379    leave out
        'ToxCast',  # 307282    leave out
        'PAMPA_NCATS',  # 408
        'HIA_Hou',  # 117
        'BBB_Martins',  # 406
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

@dataclass
class TrainConfig:
    model_name: str = "openai/gpt-oss-20b"
    lora_rank: int = 32
    learning_rate: float = 1e-5
    batch_size: int = 32
    group_size: int = 8
    max_tokens: int = 16384
    max_turns: int = 40
    format_bonus: float = 0
    save_every: int = 44
    eval_every: int = 11
    eval_max_samples: int = 1000
    eval_temperature: float = 1
    eval_n_samples: int = 1
    eval_max_retries: int = 4
    use_tools: bool = True
    wandb_project: str = "tdc-tinker-grpo"
    wandb_run_id: Optional[str] = None #"hbnj1kto"
    wandb_name: Optional[str] = "Naive GRPO DILI 3 epochs"
    data_dir: Path = PROJECT_ROOT / "DataPrepare/TDC_train_prompts_label_scaffold"
    log_path: str = PROJECT_ROOT / "logs/tinker/DILI/3-epochs-naive-grpo"
    exclude_tasks: tuple = EXCLUDED_TASKS
    resume_from: Optional[str] = None # "tinker://be372b79-3da0-589d-b696-ebb65c3a86ec:train:0/weights/step_000470"
    resume_step: int = 470
    data_seed: int = 1
    rollout_save_dir: Optional[str] = PROJECT_ROOT / "train/RL/Tinker/rollouts/DILI/3-epochs-naive-grpo"
    epochs: int = 3

@dataclass
class EvalConfig:
    test_data_dir: Path = PROJECT_ROOT / "DataPrepare" / "TDC_valid_prompts_label_scaffold"
    eval_tasks: Optional[list[str]] = None
    skip_tasks: tuple = EXCLUDED_TASKS
    max_tokens: int = 16384
    max_turns: int = 40
    temperature: float = 0.6
    top_p: float = 0.95
    n_samples: int = 1
    use_tools: bool = True
    max_samples_per_task: Optional[int] = None
    log_dir: Optional[str] = None
    verbose: bool = True
    eval_max_retries: int = 4

cfg = TrainConfig()

# Global Constants
MAX_CONTEXT_TOKENS = 120_000
SAMPLE_TIMEOUT_SEC = 900
STOP_TOKEN_IDS = [200012, 200002]  # <|call|>, <|return|>
