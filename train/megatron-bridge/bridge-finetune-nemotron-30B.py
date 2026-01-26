import os
import argparse
# for k in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
#     os.environ.pop(k, None)        # 禁用 Tri Dao flash-attn

from megatron.bridge import AutoBridge
# import megatron.bridge.recipes.qwen.qwen3_235b_a22b as qwen_a22b
from megatron.bridge.training.config import ConfigContainer, FinetuningDatasetConfig, MockGPTDatasetConfig, CheckpointConfig, TrainingConfig, OptimizerConfig, SchedulerConfig, LoggerConfig, TokenizerConfig, DistributedInitConfig, DistributedDataParallelConfig
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig
from megatron.bridge.training.finetune import finetune
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.models.decorators import torchrun_main
from transformers import AutoModelForImageTextToText, AutoModelForCausalLM, AutoConfig, AutoTokenizer
from pathlib import Path
import torch
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.initialize import destroy_global_state
from megatron.core.transformer.enums import AttnBackend
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing

from megatron.bridge.recipes.gpt_oss import gpt_oss_20b_pretrain_config

current_dir = Path(__file__).parent.resolve()

def setup_dist():
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",   # torchrun 会提供 MASTER_ADDR/PORT 等
        )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

def count_train_valid_test_samples(data_root):
    sample_count_dict = {"training.jsonl": 0,
                         "validation.jsonl": 0,
                         "test.jsonl": 0}
    split_path = data_root
    if split_path.exists():
        for file in split_path.iterdir():
            if file.suffix == ".jsonl" and file.name in sample_count_dict:
                with open(file, "r") as f:
                    sample_count_dict[file.name] = sum(1 for _ in f)
                print(f"{file.name}: {sample_count_dict[file.name]} lines")
    return sample_count_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Megatron Bridge Fine-tuning Script")

    # Model paths
    parser.add_argument("--hf_model_save_dir", type=str,
                        default=str(current_dir.parent / "checkpoints" / "megatron" / "hf_version" / "intern_s1_text_llm"),
                        help="HuggingFace model save directory")
    parser.add_argument("--megatron_model_save_dir", type=str,
                        default=str(current_dir.parent.parent / "checkpoints" / "megatron" / "megatron_version" / "gpt-oss-20b" / 'TDC_SFT_data_binary_Scaffold_wo_herg-c_ToxCast_butkiewicz'),
                        help="Megatron model save directory")
    parser.add_argument("--save_megatron_model", action="store_true",
                        help="Whether to save the Megatron model")
    # Data paths
    parser.add_argument("--data_root", type=str,
                        default=str(current_dir.parent.parent / 'DataPrepare/SFT_data/SFT_data/GPT/TDC_SFT_data_binary_Scaffold_wo_herg-c_ToxCast_butkiewicz'),
                        help="Root directory for training data")

    # Model configuration
    parser.add_argument("--moe_model", action="store_true", default=True,
                        help="Whether to use MoE model")
    parser.add_argument("--seq_length", type=int, default=2048,
                        help="Sequence length")
    parser.add_argument("--attn_backend", type=str, default="auto", choices=["auto", "fused", "flash"],
                        help="Attention backend to use")

    # Parallelism configuration
    parser.add_argument("--tp", type=int, default=None,
                        help="Tensor model parallel size (default: 4 for MoE, 1 otherwise)")
    parser.add_argument("--pp", type=int, default=1,
                        help="Pipeline model parallel size")
    parser.add_argument("--ep", type=int, default=8,
                        help="Expert model parallel size (default: 4 for MoE, 1 otherwise)")
    parser.add_argument("--etp", type=int, default=1,
                        help="Expert tensor parallel size")

    # Training configuration
    parser.add_argument("--micro_batch_size", type=int, default=1,
                        help="Micro batch size")
    parser.add_argument("--global_batch_size", type=int, default=8,
                        help="Global batch size")
    parser.add_argument("--train_iters", type=int, default=5,
                        help="Number of training iterations")
    parser.add_argument("--eval_iters", type=int, default=1,
                        help="Number of evaluation iterations")
    parser.add_argument("--eval_interval", type=int, default=5,
                        help="Evaluation interval")

    # Optimizer configuration
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5,
                        help="Minimum learning rate")
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="Optimizer type")

    # LoRA configuration
    parser.add_argument("--lora_dim", type=int, default=16,
                        help="LoRA rank r")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")

    # Tokenizer configuration
    parser.add_argument("--tokenizer_model", type=str, default="internlm/Intern-S1-mini",
                        help="Tokenizer model name")

    return parser.parse_args()


@torchrun_main
def main():
    # Parse command line arguments
    args = parse_args()

    # setup_dist()
    print("NVTE_FLASH_ATTN =", os.environ.get("NVTE_FLASH_ATTN"))
    print("NVTE_FUSED_ATTN =", os.environ.get("NVTE_FUSED_ATTN"))
    print("NVTE_UNFUSED_ATTN =", os.environ.get("NVTE_UNFUSED_ATTN"))

    sample_count_dict = count_train_valid_test_samples(Path(args.data_root))
    train_samples = sample_count_dict['training.jsonl']
    # valid_samples = sample_count_dict['valid']
    # test_samples = sample_count_dict['test']

    one_epoch_iters = train_samples // args.global_batch_size

    # logger_cfg = LoggerConfig(
    #     log_interval=1, 
    #     # tensorboard_dir=tensorboard_dir, 
    #     # log_timers_to_tensorboard=True, 
    #     wandb_project='gpt-oss-tdc', 
    #     wandb_entity='reasonv', 
    #     wandb_exp_name='gpt-oss-tdc-sft', 
    # )

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=int(one_epoch_iters * 0.05),
        lr_decay_iters=int(one_epoch_iters * 0.95),
        max_lr=5e-6,
        min_lr=0.0,
    )
    
    # args.train_iters = one_epoch_iters

    # data_cfg = MockGPTDatasetConfig(random_seed=42,
    #                                 sequence_length=1024,
    #                                 reset_position_ids=False,
    #                                 reset_attention_mask=False,
    #                                 create_attention_mask=True,  # TODO: 如果用 TE/FlashAttention 的注意力核，应该就不用这个了？
    #                                 eod_mask_loss=True,
    #                                 dataloader_type='batch',  # for finetune (SFT)
    #                                 )
    data_cfg = FinetuningDatasetConfig(
        dataset_root=args.data_root,
        seq_length=args.seq_length,
        dataloader_type="batch",
        do_validation=False,
        do_test=False,
        memmap_workers=20,
        dataset_kwargs=dict(
            answer_only_loss=True,
            chat=False,  # 打开 chat 数据集
            use_hf_tokenizer_chat_template=False,  # 用 HF chat_template 渲染
            label_key="output",  # 不用也行，chat 模式按最后一轮 assistant 计损
            truncation_field="input",  # 截断对话内容
            get_attention_mask_from_fusion=True,
            add_eos=False,
            # num_workers=120,
        ),
        # packed_sequence_specs=PackedSequenceSpecs(
        #     packed_sequence_size=seq_length,
        #     tokenizer_model_name="internlm/Intern-S1-mini",
        #     # packed_train_data_path=pack_root / "training_1024.npy",
        #     # packed_val_data_path=pack_root / "validation_1024.npy",
        #     # packed_metadata_path=pack_root / "packed_meta_1024.json",
        # )
    )
    kwargs = dict(
        pipeline_model_parallel_size=args.pp,
        expert_model_parallel_size=args.ep,

        dir=args.megatron_model_save_dir,
        use_null_tokenizer=False,
    )
    config = gpt_oss_20b_pretrain_config(dataset=data_cfg, **kwargs)
    attn_mode = AttnBackend.unfused
    config.model.attention_backend = attn_mode  # 可能会支持 FP8 (fused) / auto 用

    # config.logger.use_wandb = True
    config.logger.wandb_project = 'gpt-oss-tdc'
    config.logger.wandb_entity = 'reasonv'
    config.logger.wandb_exp_name = 'gpt-oss-tdc-sft'
    config.logger.log_interval = 1
    config.logger.log_timers_to_tensorboard = True

    config.optimizer = opt_config
    config.scheduler = scheduler

    config.train.train_iters = one_epoch_iters
    config.train.global_batch_size = args.global_batch_size
    # config.train.eval_iters = args.eval_iters
    # config.train.eval_interval = args.eval_interval

    config.checkpoint.save_interval = one_epoch_iters // 2 + 2


    config.model.seq_length = args.seq_length  # TODO: 注意这里要和 dataset 的长度一样


    # destroy_global_state()  # 防止 Rerun state 出错
    finetune(config, forward_step_func=forward_step)   # 开跑

if __name__ == "__main__":
    main()