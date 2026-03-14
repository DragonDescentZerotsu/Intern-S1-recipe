import asyncio
import json
import logging
import os
import random
import time
import wandb
import sys
import math
from config import cfg, TINKER_TO_HF, EvalConfig, PROJECT_ROOT
sys.path.append(str(PROJECT_ROOT))  # to import utils and tools

import numpy as np
import torch
import tinker
from tinker import types
from tinker.types.tensor_data import TensorData

from config import cfg, TINKER_TO_HF, EvalConfig
from adapters import get_adapter, patch_chat_template
from data_utils import load_train_data
from reward import extract_answer, parse_answer
from rollout import run_batch_rollouts, validate_rollout
from evaluate import inline_eval

import nest_asyncio
# 允许在此事件循环中嵌套运行异步任务 (Allows nested execution of async tasks in this event loop)
nest_asyncio.apply()

# 设置日志格式配置 (Configure logging format)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _sample_key(sample):
    return id(sample)


def _group_reward_stats(rollouts):
    rews = [r["reward"] for r in rollouts]
    mu = sum(rews) / len(rews)
    std = float(np.std(rews))
    return rews, mu, std


def _is_easy_zero_adv(mu, std):
    return std < 1e-8 and mu > 0.5


def get_scheduled_lr(progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    if cfg.lr_schedule == "none":
        return cfg.learning_rate
    if cfg.lr_schedule == "cosine_decay":
        min_lr = cfg.learning_rate * cfg.min_learning_rate_ratio
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (cfg.learning_rate - min_lr) * cosine
    raise ValueError(f"Unsupported lr_schedule: {cfg.lr_schedule}")


def maybe_save_best_checkpoint(tc, bi, eval_metrics, best_eval):
    macro_f1 = float(eval_metrics.get("eval/macro_f1", 0.0))
    if macro_f1 <= best_eval["f1"]:
        return best_eval, None

    f1_tag = f"{macro_f1:.4f}".replace(".", "p")
    ckpt_name = f"best_eval_f1_{f1_tag}_step_{bi:06d}"
    state_path = tc.save_state(name=ckpt_name).result().path
    deploy_path = tc.save_weights_for_sampler(name=ckpt_name).result().path
    updated = {
        "f1": macro_f1,
        "step": bi,
        "name": ckpt_name,
        "state_path": state_path,
        "deploy_path": deploy_path,
    }
    return updated, updated


def collect_effective_groups(adapter, samp_client, batch, sparams, sample_pool, next_sample_idx):
    """
    Retry zero-advantage groups until we reach the target effective group count
    or exhaust the per-sample rollout budget, refilling from later samples when
    a prompt still cannot produce a useful group.
    """
    target_groups = cfg.target_effective_groups or len(batch)
    max_rounds = max(cfg.max_group_rollout_rounds, 1)

    pending = list(batch)
    attempts = {_sample_key(sample): 0 for sample in batch}
    effective_pairs = []
    exhausted_pairs = []
    total_attempted_groups = 0
    total_rollout_rounds = 0
    refill_groups = 0
    easy_zero_adv_groups = 0
    hard_zero_adv_groups = 0

    while len(effective_pairs) < target_groups:
        if not pending:
            need = target_groups - len(effective_pairs)
            refill = sample_pool[next_sample_idx : next_sample_idx + need]
            next_sample_idx += len(refill)
            refill_groups += len(refill)
            for sample in refill:
                attempts[_sample_key(sample)] = 0
            pending = list(refill)
            if not pending:
                break

        total_rollout_rounds += 1
        total_attempted_groups += len(pending)
        paired = asyncio.run(run_batch_rollouts(adapter, samp_client, pending, sparams))
        paired_by_key = {_sample_key(sample): (sample, rollouts) for sample, rollouts in paired}

        next_pending = []
        for sample in pending:
            key = _sample_key(sample)
            attempts[key] += 1
            pair = paired_by_key.get(key)

            if pair is None:
                if attempts[key] < max_rounds:
                    next_pending.append(sample)
                else:
                    exhausted_pairs.append((sample, []))
                continue

            _, rollouts = pair
            _, mu, std = _group_reward_stats(rollouts)
            if _is_easy_zero_adv(mu, std):
                easy_zero_adv_groups += 1
                exhausted_pairs.append((sample, rollouts))
            elif std < 1e-8 and attempts[key] < max_rounds:
                hard_zero_adv_groups += 1
                next_pending.append(sample)
            elif std < 1e-8:
                hard_zero_adv_groups += 1
                exhausted_pairs.append((sample, rollouts))
            else:
                effective_pairs.append((sample, rollouts))

        pending = next_pending

    paired = effective_pairs + exhausted_pairs
    retry_groups = sum(max(v - 1, 0) for v in attempts.values())
    return paired, next_sample_idx, {
        "target_groups": target_groups,
        "effective_groups": len(effective_pairs),
        "exhausted_groups": len(exhausted_pairs),
        "refill_groups": refill_groups,
        "easy_zero_adv_groups": easy_zero_adv_groups,
        "hard_zero_adv_groups": hard_zero_adv_groups,
        "retry_groups": retry_groups,
        "rollout_rounds": total_rollout_rounds,
        "attempted_groups": total_attempted_groups,
    }

def main():
    """
    Main Training Loop for GRPO (Grounded Reward Policy Optimization) with Tinker.
    Tinker 的主训练循环，负责初始化环境、启动采样和执行模型权重的更新。
    """
    os.makedirs(cfg.log_path, exist_ok=True)
    from transformers import AutoTokenizer

    # 根据配置加载对应的 HF Tokenizer 并针对部分模型结构做特殊魔改修复 
    # (Load the corresponding HF Tokenizer base on configurations and patch chat templates for specific model structures)
    hf_name = TINKER_TO_HF.get(cfg.model_name, cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    tokenizer = patch_chat_template(tokenizer)

    # 实例化处理特定模型结构生成、思考过程抓取的适配器 
    # (Instantiate an adapter to handle generation and thought-process extraction specific to the model structure)
    adapter = get_adapter(cfg.model_name, tokenizer)
    from config import STOP_TOKEN_IDS
    stop = STOP_TOKEN_IDS
    logger.info(f"Model: {cfg.model_name} (HF: {hf_name}), Adapter: {type(adapter).__name__}, Stop: {stop}")

    # 加载和打乱 TDC 训练数据集 (Load and shuffle the TDC training dataset)
    random.seed(cfg.data_seed)
    all_samples = load_train_data(cfg.data_dir, cfg.exclude_tasks, getattr(cfg, "task", None), getattr(cfg, "playbook_dir", None))
    random.shuffle(all_samples)
    easy_samples_prompts = set()
    
    # 训练轮数按“将当前样本池完整走完一次”来定义
    # (Define an epoch as traversing the current sample pool once.)
    initial_sample_count = len(all_samples)
    nominal_n_batches = math.ceil(initial_sample_count / max(cfg.batch_size, 1)) * cfg.epochs
    logger.info(
        f"Samples: {initial_sample_count} | Target epochs: {cfg.epochs} "
        f"(nominal steps≈{nominal_n_batches})"
    )

    # 实例化 Tinker 的核心服务客户端并创建一个专用于梯度更新的 LoRA 训练服务端
    # (Instantiate Tinker's core ServiceClient and create a LoRA training client dedicated to gradient updates)
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=cfg.model_name, rank=cfg.lora_rank)

    start_batch = 0
    # 处理断点续训 (Handle resuming from checkpoints)
    if cfg.resume_from:
        logger.info(f"Resuming from checkpoint: {cfg.resume_from}, step {cfg.resume_step}")
        tc.load_state(path=cfg.resume_from).result()
        start_batch = cfg.resume_step
        logger.info(f"Loaded checkpoint, resuming from batch {start_batch}")

    # 定义模型采样配置参数以及 Adam 优化器参数 (Define model sampling configs and Adam optimizer parameters)
    sparams = tinker.types.SamplingParams(max_tokens=cfg.max_tokens, stop=STOP_TOKEN_IDS, temperature=1.0)
    # 初始化 W&B 实验看板面板记录 (Initialize W&B experimental dashboard logging)
    run_name = cfg.wandb_name or f"{cfg.model_name.split('/')[-1]}_lr{cfg.learning_rate}_bs{cfg.batch_size}_g{cfg.group_size}"
    wandb.init(
        project=cfg.wandb_project,
        name=run_name,
        id=cfg.wandb_run_id,
        resume="allow" if cfg.wandb_run_id else None,
        config={
            k: getattr(cfg, k) for k in [
                "model_name","lora_rank","learning_rate","batch_size","group_size",
                "lr_schedule","min_learning_rate_ratio",
                "target_effective_groups","max_group_rollout_rounds",
                "checkpoint_strategy","save_final_state",
                "max_tokens","max_turns","reward_format_bonus","reward_use_tools","eval_every","eval_max_samples",
                "data_seed","resume_from","resume_step","filter_easy_samples","easy_sample_retry_prob",
            ]
          } | {"n_samples": len(all_samples),
              "nominal_n_batches": nominal_n_batches, "initial_sample_count": initial_sample_count, "epochs": cfg.epochs,
              "adapter": type(adapter).__name__, "hf_name": hf_name},
    )

    mf = open(os.path.join(cfg.log_path, "metrics.jsonl"), "a")
    best_eval = {"f1": float("-inf"), "step": None, "state_path": None, "deploy_path": None}

    # 进入主要批次式前向训练主循环 (Enter main batched forward training loop)
    bi = start_batch
    epoch_round = min(start_batch // max(math.ceil(initial_sample_count / max(cfg.batch_size, 1)), 1), cfg.epochs)
    epoch_cursor = 0
    total_seen_samples = start_batch * cfg.batch_size

    while epoch_round < cfg.epochs:
        t_batch_start = time.time()
        epoch = epoch_round

        # 在当前样本池耗尽后进入下一轮重新采样/过滤
        # (Once the current sample pool is exhausted, start a new reshuffled round.)
        if epoch_cursor >= len(all_samples):
          epoch_round += 1
          if epoch_round >= cfg.epochs:
              break
          random.seed(cfg.data_seed + epoch_round)
          if getattr(cfg, "filter_easy_samples", False) and easy_samples_prompts:
              old_len = len(all_samples)
              retained_easy = 0
              filtered_easy = 0
              next_samples = []
              for s in all_samples:
                  prompt_text = s.get("text", "")
                  if prompt_text not in easy_samples_prompts:
                      next_samples.append(s)
                      continue
                  if random.random() < cfg.easy_sample_retry_prob:
                      next_samples.append(s)
                      retained_easy += 1
                  else:
                      filtered_easy += 1
              all_samples = next_samples
              logger.info(
                  f"Epoch round {epoch_round}: Easy sample retry prob={cfg.easy_sample_retry_prob:.0%}. "
                  f"Retried {retained_easy}, filtered {filtered_easy}. Retained {len(all_samples)} / {old_len}"
              )
          random.shuffle(all_samples)
          epoch_cursor = 0
          logger.info(f"=== EPOCH ROUND {epoch_round} === (reshuffled)")

        if not all_samples:
            logger.warning("No samples available after filtering; stopping training early.")
            break

        # ── CHECKPOINT MODEL (保存模型状态断点) ─────────────────────────────────────────
        t_ckpt_start = time.time()
        if cfg.checkpoint_strategy == "interval" and cfg.save_every > 0 and bi > 0 and bi % cfg.save_every == 0:
            p = tc.save_state(name=f"step_{bi:06d}").result().path
            logger.info(f"{'='*60}")
            logger.info(f"CHECKPOINT SAVED: {p}")
            logger.info(f"{'='*60}")
            with open(os.path.join(cfg.log_path, "checkpoints.txt"), "a") as cf:
                cf.write(f"step={bi} path={p}\n")
        t_ckpt = time.time() - t_ckpt_start

        # ── SAMPLER SETUP (设定当前周期的采样器权重) ─────────────────────────────────────
        # 通过 epoch 内游标取样；如果需要补有效 group，后续会继续消耗后面的样本
        # (Take the next batch from an epoch cursor; top-ups consume later samples too.)
        step_cursor_start = epoch_cursor
        batch = all_samples[epoch_cursor : epoch_cursor + cfg.batch_size]
        epoch_cursor += len(batch)
        
        if not batch:
            epoch_cursor = len(all_samples)
            continue

        t_sampler_start = time.time()
        
        # 将训练客户端最新的权重落盘，从而交给采样器推理 
        # (Save the latest weights from the training client to disk to hand them over to the sampler for inference)
        spath = tc.save_weights_for_sampler(name=f"step_{bi:06d}").result().path
        samp_client = sc.create_sampling_client(model_path=spath)
        t_sampler = time.time() - t_sampler_start

        # ── ROLLOUTS (并行进行 MCTS 生成采样获取环境交互数据以供强化) ───────────────
        t_rollout_start = time.time()
        # 反复为 zero-adv group 重采样，直到凑够目标有效 group 数或达到每个样本的最大重试轮数
        # (Retry zero-adv groups until we hit the target effective group count or the per-sample retry budget;
        #  if still short, pull fresh samples from later in the epoch.)
        paired, epoch_cursor, rollout_stats = collect_effective_groups(
            adapter, samp_client, batch, sparams, all_samples, epoch_cursor
        )
        t_rollout = time.time() - t_rollout_start
        step_seen_samples = max(epoch_cursor - step_cursor_start, 0)
        total_seen_samples += step_seen_samples
        consumed_samples = min(epoch_cursor, len(all_samples))
        data_epoch_progress = consumed_samples / max(len(all_samples), 1)
        data_progress_total = min((epoch_round + data_epoch_progress) / max(cfg.epochs, 1), 1.0)

        # ── ROLLOUT VALIDATION (轨迹数据合法性检查/抽查) ─────────────────────────────────
        if bi < 3 or bi % 10 == 0:
            n_validated = 0
            n_failed = 0
            for sample, rollouts in paired[:5]:
                for ro in rollouts[:2]:
                    report = validate_rollout(adapter, ro, sample)
                    n_validated += 1
                    # 检查是不是遇到了断言失败或者采样崩溃，例如 token_length 对不上的情况
                    # (Checks if we've encountered assertions failures or sampling crashes like mismatched token lengths)
                    if not report["valid"]:
                        n_failed += 1
                        logger.error(f"ROLLOUT VALIDATION FAILED: {report['errors']}")
                        logger.error(f"  task={sample['task_name']} gt={sample['Y']} "
                                    f"reward={report['reward']} "
                                    f"turns={report['n_turns']} "
                                    f"has_lps={report['has_logprobs']}")
                    elif bi < 3:
                        logger.info(f"  VALID: {sample['task_name']} | "
                                    f"turns={report['n_turns']} | "
                                    f"has_lps={report['has_logprobs']} | "
                                    f"r={report['reward']}")
            if n_failed > 0:
                logger.error(f"VALIDATION: {n_failed}/{n_validated} rollouts failed")


        # ── DIAGNOSTICS COLLECTION (计算 Advantage 优势与组装梯度数据 Datum) ─────────
        t_diag_start = time.time()
        all_datums = []
        batch_rewards = []
        task_rewards = {}
        skipped = 0
        skipped_easy = 0
        skipped_hard = 0
        class_rewards = {0: [], 1: []}
        predictions = []
        null_preds = 0
        format_correct_count = 0
        total_rollouts = 0
        response_token_counts = []
        seq_len_counts = []
        tool_call_counts = []
        sample_log = []
        expected_rollouts = rollout_stats["attempted_groups"] * cfg.group_size
        actual_rollouts = sum(len(g) for _, g in paired)

        # 遍历当前批次每一个 Prompt 的不同重采样轨迹组
        # (Iterate through every prompt's different resampling trajectory groups in the current batch)
        for sample, rollouts in paired:
            tn = sample["task_name"]
            gt = int(sample["Y"]) # Ground-truth Label
            rews = [r["reward"] for r in rollouts]
            mu = sum(rews) / len(rews) # 组内奖赏均值 (Group mean reward)
            std = float(np.std(rews)) # 组内奖赏标准差 (Group reward standard dev)
            batch_rewards.append(mu)
            task_rewards.setdefault(tn, []).append(mu)
            class_rewards[gt].append(mu)

            # 更新一些实验监控使用的日志数据并计算该 prompt 下是否有可用回答
            # (Update experimental dashboard logs and check if the prompt yielded a valid parseable answer)
            for ro in rollouts:
                total_rollouts += 1
                final_text = adapter.extract_final_text(ro["response_text"])
                ans_text, fmt = extract_answer(final_text)
                pred = parse_answer(ans_text, fmt)

                if fmt:
                    format_correct_count += 1
                if pred is None:
                    null_preds += 1
                else:
                    predictions.append((pred, gt))

                # 记录采样回复产生的 Token 总计开销用于评估推理速度与花费
                # (Record the token cost of sampled responses to evaluate speed and length constraints)
                total_gen = sum(len(td["gen_tokens"]) for td in ro["turn_data"])
                total_seq = sum(len(td["prompt_tokens"]) + len(td["gen_tokens"]) for td in ro["turn_data"])
                response_token_counts.append(total_gen)
                seq_len_counts.append(total_seq)
                tool_call_counts.append(max(len(ro["turn_data"]) - 1, 0))  # 额外的轮次为发生了外部工具调用 (extra turns signify external tool use occurrences)

            # GRPO 核心思想：假如对于一个问题产生的所有几条回答的奖赏完全一致（std -> 0），
            # 那么不产生可供利用的策略优化空间，直接跳过。
            # (Core GRPO idea: If standard dev of rewards is extremely close to 0, it produces no advantage optimization gradients space, so we skip it)
            if std < 1e-8:
                skipped += 1
                if mu > 0.5:
                    skipped_easy += 1
                    if getattr(cfg, "filter_easy_samples", False):
                        easy_samples_prompts.add(sample.get("text", ""))
                else:
                    skipped_hard += 1
                    easy_samples_prompts.discard(sample.get("text", ""))
                continue
            easy_samples_prompts.discard(sample.get("text", ""))

            # 使用奖赏计算 Advantage = (R - μ) / σ 来归一化优势数值
            # (Normalize Advantage using (R - μ) / σ)
            for ro in rollouts:
                adv = (ro["reward"] - mu) / std
                if adv == 0.0:
                    continue

                for td in ro["turn_data"]:
                    prompt_tokens = td["prompt_tokens"]
                    gen_tokens = td["gen_tokens"]
                    gen_lps = td["gen_logprobs"]

                    # 组装 Tinker API 所需的训练输入序列，Target 必须错开 1 个位置
                    # (Assemble training input array required by Tinker API, shift target sequence by 1 index)
                    all_tokens = prompt_tokens + gen_tokens
                    input_tokens = [int(t) for t in all_tokens[:-1]]
                    target_tokens = all_tokens[1:]

                    ob_len = len(prompt_tokens) - 1
                    gen_len = len(gen_tokens)
                    seq_len = len(input_tokens)

                    # 对观察/前缀 (Prompt) 部分执行 Loss 遮罩(Mask 填 0)，并将 Advantage 填入生成的序列部分
                    # (Apply loss padding/mask-zeros to the observation prefix part, and fill in the Advantage to the generated seq suffix)
                    padded_logprobs = [0.0] * ob_len + gen_lps
                    padded_logprobs = (padded_logprobs + [0.0] * seq_len)[:seq_len]

                    padded_advantages = [0.0] * ob_len + [adv] * gen_len
                    padded_advantages = (padded_advantages + [0.0] * seq_len)[:seq_len]

                    assert len(input_tokens) == len(target_tokens) == len(padded_logprobs) == len(padded_advantages), (
                        f"Datum alignment: {len(input_tokens)}, {len(target_tokens)}, "
                        f"{len(padded_logprobs)}, {len(padded_advantages)}"
                    )
                    
                    # 组装为可派发的张量传输结构
                    # (Assemble into dispatchable tensor format structure)
                    all_datums.append(types.Datum(
                        model_input=types.ModelInput.from_ints(tokens=input_tokens),
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                            "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                            "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                        },
                    ))

        # ── SAMPLE LOGGING (将少数具体对话案例样本提取以便人工检查观测模型表现) ─────
        if paired:
            for sample, rollouts in paired[:10]:
                ro = rollouts[0]
                convo_lines = []
                for msg in ro.get("messages", []):
                    role = msg.get("role", "?")
                    if role == "system":
                        convo_lines.append(f"[SYSTEM] {msg.get('content', '')[:200]}...")
                    elif role == "user":
                        convo_lines.append(f"[USER] {msg.get('content', '')}")
                    elif role == "assistant":
                        thinking = msg.get("thinking", "")
                        content = msg.get("content", "")
                        tc_list = msg.get("tool_calls", [])
                        if thinking:
                            convo_lines.append(f"[ASSISTANT thinking] {thinking}")
                        if tc_list:
                            for tc_item in tc_list:
                                fn = tc_item.get("function", {})
                                convo_lines.append(f"[TOOL CALL] {fn.get('name', '?')}({fn.get('arguments', '')})")
                        if content:
                            convo_lines.append(f"[ASSISTANT] {content}")
                    elif role == "tool":
                        convo_lines.append(f"[TOOL RESULT {msg.get('name', '?')}] {msg.get('content', '')}")
                full_convo = "\n".join(convo_lines)

                sample_log.append({
                    "task": sample["task_name"],
                    "gt": int(sample["Y"]),
                    "prompt_text": sample.get("text", ""),
                    "reward": ro["reward"],
                    "response": ro["response_text"],
                    "final": adapter.extract_final_text(ro["response_text"]),
                    "conversation": full_convo,
                    "n_turns": sum(1 for m in ro.get("messages", []) if m.get("role") == "assistant"),
                })
        t_diag = time.time() - t_diag_start

        # ── SAVE ROLLOUTS (本地持久化保存对话轨迹结果，供复盘和离线分析) ─────────────
        if cfg.rollout_save_dir and paired:
            os.makedirs(cfg.rollout_save_dir, exist_ok=True)
            rollout_records = []
            for sample, rollouts in paired:
                for ri, ro in enumerate(rollouts):
                    # Build readable conversation
                    convo_lines = []
                    for msg in ro.get("messages", []):
                        role = msg.get("role", "?")
                        if role == "system":
                            convo_lines.append(f"[SYSTEM] {msg.get('content', '')[:200]}...")
                        elif role == "user":
                            convo_lines.append(f"[USER] {msg.get('content', '')}")
                        elif role == "assistant":
                            thinking = msg.get("thinking", "")
                            content = msg.get("content", "")
                            tc_list = msg.get("tool_calls", [])
                            if thinking:
                                convo_lines.append(f"[ASSISTANT thinking] {thinking}")
                            if tc_list:
                                for tc_item in tc_list:
                                    fn = tc_item.get("function", {})
                                    convo_lines.append(f"[TOOL CALL] {fn.get('name', '?')}({fn.get('arguments', '')})")
                            if content:
                                convo_lines.append(f"[ASSISTANT] {content}")
                        elif role == "tool":
                            convo_lines.append(f"[TOOL RESULT {msg.get('name', '?')}] {msg.get('content', '')}")

                    rollout_records.append({
                        "batch": bi,
                        "task": sample["task_name"],
                        "gt": int(sample["Y"]),
                        "rollout_idx": ri,
                        "reward": ro["reward"],
                        "prompt_text": sample.get("text", ""),
                        "response_text": ro["response_text"],
                        "final_text": adapter.extract_final_text(ro["response_text"]),
                        "conversation": "\n".join(convo_lines),
                        "n_turns": len(ro["turn_data"]),
                        "n_tokens": sum(len(td["prompt_tokens"]) + len(td["gen_tokens"]) for td in ro["turn_data"]),
                        "n_response_tokens": sum(len(td["gen_tokens"]) for td in ro["turn_data"]),
                        "messages": ro.get("messages", []),
                    })

            fp = os.path.join(cfg.rollout_save_dir, f"batch_{bi:06d}.jsonl")
            with open(fp, "w") as rf:
                for rec in rollout_records:
                    rf.write(json.dumps(rec, default=str) + "\n")

        # ── TRAIN STEP (使用构造好的 Datum 在 Tinker 后台使用策略梯度更新模型网络) ──
        t_train_start = time.time()
        current_lr = get_scheduled_lr(data_progress_total)
        adam = types.AdamParams(learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8)
        if all_datums:
            # 传输数据给到 LoRA 后端计算向后损失传递 (Forward all datums to LoRA backend to compute backward loss propagation)
            fb = tc.forward_backward(all_datums, loss_fn="importance_sampling")
            # 发起一步 Adam 优化更新权重 (Execute an Adam optimization step that updates the weights)
            os_ = tc.optim_step(adam)
            fb.result(); os_.result()
        else:
            logger.warning(f"Batch {bi}: no datums")
        t_train = time.time() - t_train_start

        # ── METRICS (汇总本批次所有参数日志如训练耗时到 W&B 看板) ───────────────────
        t_total = time.time() - t_batch_start
        mr = sum(batch_rewards) / max(len(batch_rewards), 1)

        n_pred_a = sum(1 for p, _ in predictions if p == 0)
        n_pred_b = sum(1 for p, _ in predictions if p == 1)

        eta_hours = 0.0
        if data_progress_total > 0:
            elapsed_proxy = t_total / max(data_epoch_progress if data_epoch_progress > 0 else 1e-6, 1e-6)
            remaining_epochs = max(cfg.epochs - (epoch_round + data_epoch_progress), 0.0)
            eta_hours = elapsed_proxy * remaining_epochs / 3600

        metrics = {
            "batch": bi,
            "epoch": epoch,
            "progress": data_progress_total,
            "data/consumed_samples_epoch": consumed_samples,
            "data/epoch_progress": data_epoch_progress,
            "data/consumed_samples_total": total_seen_samples,
            "data/target_samples_total": initial_sample_count * cfg.epochs,
            "data/epoch_equivalent": epoch_round + data_epoch_progress,
            "data/total_progress": data_progress_total,
            # 时间耗时指标 (Timing)
            "time/total_sec": t_total,
            "time/checkpoint_sec": t_ckpt,
            "time/sampler_setup_sec": t_sampler,
            "time/rollout_sec": t_rollout,
            "time/diagnostics_sec": t_diag,
            "time/train_step_sec": t_train,
            "time/rollout_pct": t_rollout / max(t_total, 0.01) * 100,
            "time/train_pct": t_train / max(t_total, 0.01) * 100,
            "time/overhead_pct": (t_ckpt + t_sampler + t_diag) / max(t_total, 0.01) * 100,
            "time/sec_per_rollout": t_rollout / max(actual_rollouts, 1),
            "time/eta_hours": eta_hours,
            "train/learning_rate": current_lr,
            # 模型强化奖赏信号分布表现 (Reward)
            "reward/mean": mr,
            "reward/std": float(np.std(batch_rewards)) if batch_rewards else 0,
            "reward/min": min(batch_rewards) if batch_rewards else 0,
            "reward/max": max(batch_rewards) if batch_rewards else 0,
            "reward/n_prompts": len(batch_rewards),
            "reward/class_0_mean": sum(class_rewards[0]) / max(len(class_rewards[0]), 1),
            "reward/class_1_mean": sum(class_rewards[1]) / max(len(class_rewards[1]), 1),
            "reward/format_rate": format_correct_count / max(total_rollouts, 1),
            "reward/null_rate": null_preds / max(total_rollouts, 1),
            "reward/pred_balance_B": n_pred_b / max(n_pred_a + n_pred_b, 1),
            # VLLM 采样与对齐状态指标 (Training signal)
            "train/n_datums": len(all_datums),
            "train/skipped": skipped,
            "train/skipped_easy": skipped_easy,
            "train/skipped_hard": skipped_hard,
            "train/zero_adv_rate": skipped / max(len(paired), 1),
            "train/datums_per_prompt": len(all_datums) / max(len(paired) - skipped, 1) if len(paired) > skipped else 0,
            "train/target_effective_groups": rollout_stats["target_groups"],
            "train/effective_groups": rollout_stats["effective_groups"],
            "train/exhausted_groups": rollout_stats["exhausted_groups"],
            "train/refill_groups": rollout_stats["refill_groups"],
            "train/easy_zero_adv_groups": rollout_stats["easy_zero_adv_groups"],
            "train/hard_zero_adv_groups": rollout_stats["hard_zero_adv_groups"],
            "train/retry_groups": rollout_stats["retry_groups"],
            "train/rollout_rounds": rollout_stats["rollout_rounds"],
            # 推理和思考量表现统计 (Generation behavior)
            "gen/total_rollouts": actual_rollouts,
            "gen/failed_rollouts": expected_rollouts - actual_rollouts,
            "gen/failure_rate": (expected_rollouts - actual_rollouts) / max(expected_rollouts, 1),
            "gen/mean_response_tokens": float(np.mean(response_token_counts)) if response_token_counts else 0,
            "gen/max_response_tokens": max(response_token_counts) if response_token_counts else 0,
            "gen/mean_seq_len": float(np.mean(seq_len_counts)) if seq_len_counts else 0,
            "gen/max_seq_len": max(seq_len_counts) if seq_len_counts else 0,
            "gen/mean_tool_calls": float(np.mean(tool_call_counts)) if tool_call_counts else 0,  # TODO: Wrong!
            "gen/tool_use_rate": sum(1 for t in tool_call_counts if t > 0) / max(len(tool_call_counts), 1),
            "gen/total_tokens_generated": sum(response_token_counts),
            "gen/tokens_per_second": sum(response_token_counts) / max(t_rollout, 0.01),
        }
        for t, rs in task_rewards.items():
            metrics[f"reward/{t}"] = sum(rs) / len(rs)

        # 录入文本抽查日志 (Attach sample texts to wandb log)
        if sample_log:
            metrics["samples"] = wandb.Table(
                columns=["task", "gt", "prompt_text", "reward", "n_turns", "final_text", "conversation", "full_response"],
                data=[[s["task"], s["gt"], s["prompt_text"], s["reward"], s["n_turns"],
                       s["final"], s["conversation"], s["response"]] for s in sample_log],
            )

        logger.info(
            f"Batch {bi} | "
            f"data_epoch={consumed_samples}/{len(all_samples)} ({data_epoch_progress:.1%}) "
            f"epoch_total={epoch_round + data_epoch_progress:.2f}/{cfg.epochs} "
            f"total_progress={data_progress_total:.1%} | "
            f"reward={mr:.3f}±{metrics['reward/std']:.3f} "
            f"datums={len(all_datums)} skip={skipped} (easy={skipped_easy} hard={skipped_hard}) "
            f"zero-adv[easy={rollout_stats['easy_zero_adv_groups']} hard={rollout_stats['hard_zero_adv_groups']}] "
            f"fmt={metrics['reward/format_rate']:.0%} null={metrics['reward/null_rate']:.0%} "
            f"predB={metrics['reward/pred_balance_B']:.0%} "
            f"tools={metrics['gen/mean_tool_calls']:.1f} "
            f"fail={metrics['gen/failed_rollouts']}/{expected_rollouts} | "
            f"total={t_total:.0f}s "
            f"[rollout={t_rollout:.0f}s({metrics['time/rollout_pct']:.0f}%) "
            f"train={t_train:.0f}s({metrics['time/train_pct']:.0f}%) "
            f"setup={t_sampler:.0f}s ckpt={t_ckpt:.0f}s] | "
            f"tok/s={metrics['gen/tokens_per_second']:.0f} "
            f"ETA={metrics['time/eta_hours']:.1f}h"
        )
        wandb.log(metrics, step=bi)
        mf.write(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, wandb.Table)}) + "\n")
        mf.flush()

        # ── EVAL (运行固定周期的线上内联直接回答评估验证其真实知识留存能力) ─────────
        if cfg.eval_every > 0 and bi % cfg.eval_every == 0:
            t_eval_start = time.time()
            ecfg = EvalConfig(
                max_samples_per_task=cfg.eval_max_samples, n_samples=cfg.eval_n_samples,
                temperature=cfg.eval_temperature, log_dir=cfg.log_path, verbose=True,
                eval_max_retries=cfg.eval_max_retries,
            )
            ecfg.eval_metadata = {
                "eval/step": bi,
                "eval/epoch_nominal": epoch,
                "eval/data_consumed_samples_epoch": consumed_samples,
                "eval/data_epoch_progress": data_epoch_progress,
                "eval/data_epoch_equivalent": epoch_round + data_epoch_progress,
                "eval/data_total_progress": data_progress_total,
            }
            em = inline_eval(adapter, tc, sc, step=bi, ecfg=ecfg)
            t_eval = time.time() - t_eval_start
            em["eval/time_sec"] = t_eval
            saved_best = None
            if cfg.checkpoint_strategy == "best_eval":
                best_eval, saved_best = maybe_save_best_checkpoint(tc, bi, em, best_eval)
                em["eval/best_macro_f1"] = best_eval["f1"]
                em["eval/best_step"] = -1 if best_eval["step"] is None else best_eval["step"]
                if saved_best is not None:
                    logger.info(
                        f"New best checkpoint | step={bi} f1={saved_best['f1']:.4f} | "
                        f"state={saved_best['state_path']} deploy={saved_best['deploy_path']}"
                    )
                    with open(os.path.join(cfg.log_path, "checkpoints.txt"), "a") as cf:
                        cf.write(
                            f"best_step={bi} macro_f1={saved_best['f1']:.6f} "
                            f"state={saved_best['state_path']} deploy={saved_best['deploy_path']}\n"
                        )
            logger.info(
                f"Step {bi} eval | "
                f"data_epoch={consumed_samples}/{len(all_samples)} ({data_epoch_progress:.1%}) "
                f"total_progress={data_progress_total:.1%} | "
                f"Macro F1 = {em['eval/macro_f1']:.4f} "
                f"best={best_eval['f1']:.4f}@{best_eval['step'] if best_eval['step'] is not None else '-'} "
                f"({t_eval:.0f}s)"
            )
            wandb.log(em, step=bi)
            mf.write(json.dumps(em) + "\n"); mf.flush()
        bi += 1

    # ── FINAL (在批次迭代结束后跑最后一轮保存并彻底评估整个模型) ─────────────────────
    if cfg.save_final_state:
        fp = tc.save_state(name="final").result().path
        dp = tc.save_weights_for_sampler(name="final_deploy").result().path
        logger.info(f"Final checkpoint: {fp}\nDeploy weights: {dp}")
        with open(os.path.join(cfg.log_path, "checkpoints.txt"), "a") as cf:
            cf.write(f"step=final state={fp}\n")
            cf.write(f"step=final deploy={dp}\n")

    ecfg = EvalConfig(
        n_samples=cfg.eval_n_samples, temperature=cfg.eval_temperature,
        log_dir=os.path.join(cfg.log_path, "eval_final"), verbose=True,
        eval_max_retries=cfg.eval_max_retries,
    )
    ecfg.eval_metadata = {
        "eval/step": bi,
        "eval/epoch_nominal": cfg.epochs,
        "eval/data_consumed_samples_epoch": min(epoch_cursor, len(all_samples)),
        "eval/data_epoch_progress": min(epoch_cursor, len(all_samples)) / max(len(all_samples), 1),
        "eval/data_epoch_equivalent": min(epoch_round + (min(epoch_cursor, len(all_samples)) / max(len(all_samples), 1)), float(cfg.epochs)),
        "eval/data_consumed_samples_total": total_seen_samples,
        "eval/data_target_samples_total": initial_sample_count * cfg.epochs,
        "eval/data_total_progress": min(epoch_round / max(cfg.epochs, 1), 1.0) if epoch_cursor == 0 else min((epoch_round + min(epoch_cursor, len(all_samples)) / max(len(all_samples), 1)) / max(cfg.epochs, 1), 1.0),
    }
    em = inline_eval(adapter, tc, sc, step=bi, ecfg=ecfg)
    if cfg.checkpoint_strategy == "best_eval":
        best_eval, saved_best = maybe_save_best_checkpoint(tc, bi, em, best_eval)
        em["eval/best_macro_f1"] = best_eval["f1"]
        em["eval/best_step"] = -1 if best_eval["step"] is None else best_eval["step"]
        if saved_best is not None:
            logger.info(
                f"New best checkpoint | step={bi} f1={saved_best['f1']:.4f} | "
                f"state={saved_best['state_path']} deploy={saved_best['deploy_path']}"
            )
            with open(os.path.join(cfg.log_path, "checkpoints.txt"), "a") as cf:
                cf.write(
                    f"best_step={bi} macro_f1={saved_best['f1']:.6f} "
                    f"state={saved_best['state_path']} deploy={saved_best['deploy_path']}\n"
                )
    logger.info(
        f"Final eval | "
        f"data_epoch={min(epoch_cursor, len(all_samples))}/{len(all_samples)} "
        f"({em['eval/data_epoch_progress']:.1%}) total_progress={em['eval/data_total_progress']:.1%} | "
        f"Macro F1 = {em['eval/macro_f1']:.4f} "
        f"best={best_eval['f1']:.4f}@{best_eval['step'] if best_eval['step'] is not None else '-'}"
    )
    wandb.log(em, step=bi)

    wandb.finish(); mf.close()
    logger.info("Done.")

if __name__ == "__main__":
    main()
