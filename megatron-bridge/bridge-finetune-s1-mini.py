import os
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


@torchrun_main
def main():
    # setup_dist()
    print("NVTE_FLASH_ATTN =", os.environ.get("NVTE_FLASH_ATTN"))
    print("NVTE_FUSED_ATTN =", os.environ.get("NVTE_FUSED_ATTN"))
    print("NVTE_UNFUSED_ATTN =", os.environ.get("NVTE_UNFUSED_ATTN"))

    hf_model_save_dir = current_dir.parent/"checkpoints"/"megatron"/"hf_version"/"intern_s1_mini_text_llm"
    megatron_model_save_dir = current_dir.parent/"checkpoints"/"megatron"/"megatron_version"/"intern_s1_mini_text_llm"

    # Dataset
    data_root = current_dir.parent/'SFT_data'/'SFT_data'/'GPT_TDC_CLS'
    # pack_root = current_dir.parent/'SFT_data'/'SFT_data'/'GPT_TDC_CLS_MB_packed'

    # 统计训练集数量
    # sample_count_dict = count_train_valid_test_samples(data_root)
    # print(f"sample_count_dict = {sample_count_dict}")

    save_megatron_model = False
    attn_mode = AttnBackend.auto # AttnBackend.fused / AttnBackend.auto  可能会支持 FP8 (fused) / auto 用
    ENABLE_THINKING = False
    tp = 1
    pp = 1
    ep = 1
    etp = 1

    # 1) 从你刚导出的纯文本模型加载，并转为 Megatron 提供器
    # is_rank_0 = torch.distributed.get_rank() == 0
    bridge = AutoBridge.from_hf_pretrained(str(hf_model_save_dir), torch_dtype=torch.bfloat16)

    if megatron_model_save_dir.exists():
        # if is_rank_0:
        print("loading Megatron model from cache...")
        provider = bridge.to_megatron_provider(load_weights=False)
        provider.attention_backend = attn_mode  # 可能会支持 FP8 (fused) / auto 用
        provider.tensor_model_parallel_size = tp
        provider.pipeline_model_parallel_size = pp
        provider.pipeline_dtype = torch.bfloat16
        provider.expert_model_parallel_size = ep
        provider.expert_tensor_parallel_size = etp
        provider.finalize()
        provider.initialize_model_parallel(seed=0)
        megatron_model = bridge.load_megatron_model(
            str(megatron_model_save_dir),
            mp_overrides={
                "tensor_model_parallel_size": tp,
                "pipeline_model_parallel_size": pp,
                "expert_model_parallel_size": ep,
                "expert_tensor_parallel_size": etp,
                "pipeline_dtype": torch.bfloat16,
                "attention_backend": attn_mode,  # 这个非常重要，否则会报错 fused/auto
            },
            wrap_with_ddp=False,
        )
        megatron_model = [m.cuda() for m in megatron_model]

    else:
        # if is_rank_0:
        print("converting Megatron model from HF...")
        provider = bridge.to_megatron_provider(load_weights=True)
        provider.attention_backend = attn_mode  # fused 可能会支持 FP8 fused/auto
        provider.tensor_model_parallel_size = tp
        provider.pipeline_model_parallel_size = pp
        provider.pipeline_dtype = torch.bfloat16
        provider.expert_model_parallel_size = ep
        provider.expert_tensor_parallel_size = etp
        provider.finalize()  # 让 Bridge 按照 A22B 架构构建 Megatron Core 模型
        provider.initialize_model_parallel(seed=0)
        megatron_model = provider.provide_distributed_model(wrap_with_ddp=False)

    # if is_rank_0:
    print("\nprovider:\n",provider)
    print("\nmegatron model:\n",megatron_model)

    if not megatron_model_save_dir.exists():
        megatron_model_save_dir.mkdir(parents=True)
    if save_megatron_model or not megatron_model_save_dir.exists():
        bridge.save_megatron_model(megatron_model, str(megatron_model_save_dir))

    # see supported models
    # if is_rank_0:
    supported_models = AutoBridge.list_supported_models()

    print(f"Found {len(supported_models)} supported models:")
    for i, model in enumerate(supported_models, 1):
        print(f"  {i:2d}. {model}")

    # =========================================================================================
    # configurations
    fp8_config = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        fp8="hybrid",
        fp8_recipe="tensorwise",
        fp8_margin=0,
        fp8_amax_history_len=1024,
        fp8_amax_compute_algo="max",
        fp8_param_gather=True,
        fp8_dot_product_attention=True if attn_mode == AttnBackend.fused else False,
        fp8_param=True if attn_mode == AttnBackend.fused else False
    )

    lora_config = LoRA(
        target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        dim=16,       # LoRA rank r
        alpha=32,    # LoRA alpha
        dropout=0.05
    )

    train_cfg = TrainingConfig(micro_batch_size=1,
                               global_batch_size=4,  # TODO: 至少需要和 DP 的数量相同
                               train_iters=100,
                               eval_iters=10,
                               eval_interval=50,
                               )
    optim_cfg = OptimizerConfig(
            optimizer="adam",
            lr=3e-4,
            min_lr=1e-5,
            use_distributed_optimizer=False,
        )
    tokenizer_config = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model="internlm/Intern-S1-mini",
        hf_tokenizer_kwargs={"trust_remote_code": True,
                             # "use_fast": True
                             },
    )
    sched_cfg = SchedulerConfig(lr_warmup_fraction=0.1,
                                start_weight_decay=0.0,  # 例子：前期不施加WD
                                end_weight_decay=0.1,  # 逐步增到目标WD
                                weight_decay_incr_style="cosine",  # 或 "linear"
                                )
    # data_cfg = MockGPTDatasetConfig(random_seed=42,
    #                                 sequence_length=2048,
    #                                 reset_position_ids=False,
    #                                 reset_attention_mask=False,
    #                                 create_attention_mask=True,  # TODO: 如果用 TE/FlashAttention 的注意力核，应该就不用这个了？
    #                                 eod_mask_loss=True,
    #                                 dataloader_type='batch',  # for finetune (SFT)
    #                                 )
    data_cfg = FinetuningDatasetConfig(
        dataset_root=data_root,
        seq_length=2048,
        dataloader_type="batch",
        do_validation=True,
        do_test=True,
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
        packed_sequence_specs=PackedSequenceSpecs(
            packed_sequence_size=2048,
            tokenizer_model_name="internlm/Intern-S1-mini",
            # packed_train_data_path=pack_root / "training_1024.npy",
            # packed_val_data_path=pack_root / "validation_1024.npy",
            # packed_metadata_path=pack_root / "packed_meta_1024.json",
        )
    )

    dist_cfg = DistributedInitConfig()  # 单机默认分布式配置
    ddp_cfg = DistributedDataParallelConfig(use_distributed_optimizer=False)  # 数据并行默认配置
    logger_config = LoggerConfig()

    checkpoint_config = CheckpointConfig(
        pretrained_checkpoint=str(megatron_model_save_dir),  # 基座权重文件 in Megatron Format
        save=str(megatron_model_save_dir.parent/'lora'/'intern_s1_mini_text_llm')  # LoRA 权重保存目录 in Megatron format
    )

    config = ConfigContainer(
        train=train_cfg,
        optimizer=optim_cfg,
        scheduler=sched_cfg,
        tokenizer=tokenizer_config,
        dataset=data_cfg,
        dist=dist_cfg,
        ddp=ddp_cfg,
        model=provider,  # 注意这里 provider 其实对应的就是各种文档里的 GPTModelProvider 之类的
        mixed_precision=fp8_config,
        peft=lora_config,
        checkpoint= checkpoint_config,
        logger=logger_config,
        # ... other config parameters
    )
    config.model.seq_length = 2048  # TODO: 注意这里要和 dataset 的长度一样

    destroy_global_state()  # 防止 Rerun state 出错
    finetune(config, forward_step_func=forward_step)   # 开跑

if __name__ == "__main__":
    main()