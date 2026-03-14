import asyncio
import json
import logging
import os
import numpy as np
import tinker

from sklearn.metrics import f1_score

from config import MAX_CONTEXT_TOKENS, SAMPLE_TIMEOUT_SEC, STOP_TOKEN_IDS, TASK_TO_GROUP
from data_utils import load_test_data
from reward import extract_answer, parse_answer
from rollout import build_messages, tok_messages, execute_tools

logger = logging.getLogger(__name__)

async def _async_inference_single(adapter, sampling_client, task_name, prompt, ecfg):
    messages, tools = build_messages(adapter, task_name, prompt)
    sparams = tinker.types.SamplingParams(
        max_tokens=ecfg.max_tokens,
        stop=STOP_TOKEN_IDS,
        temperature=ecfg.temperature, top_p=ecfg.top_p,
    )
    full_resp = ""
    last_tools = set()
    for _ in range(ecfg.max_turns):
        toks = tok_messages(adapter, messages, tools)

        if len(toks) + ecfg.max_tokens > MAX_CONTEXT_TOKENS:
            logger.warning(f"Eval context {len(toks)} tok, stopping multi-turn")
            break

        try:
            fut = sampling_client.sample(
                prompt=tinker.types.ModelInput.from_ints(toks),
                num_samples=1, sampling_params=sparams,
            )
            res = await asyncio.to_thread(lambda: fut.result(timeout=SAMPLE_TIMEOUT_SEC))
        except Exception as e:
            logger.warning(f"Eval sample failed ({len(toks)} tok): {e}")
            break

        text = adapter.tokenizer.decode(
            res.sequences[0].tokens,
            skip_special_tokens=adapter.skip_special_on_decode()
        )
        for eos in adapter.eos_strips():
            if text.endswith(eos): text = text[:-len(eos)]
        full_resp += text

        calls = adapter.parse_tool_calls(text)
        if not calls:
            break
        
        current_names = set(c.get("name", "") for c in calls)
        if current_names == last_tools:
            break
        last_tools = current_names

        results = execute_tools(calls)
        messages.append(adapter.format_assistant_message(text))
        for i, tr in enumerate(results):
            messages.append(adapter.format_tool_result_message(
                tr["name"], tr["content"], call_id=f"call_{i}"
            ))

    final_text = adapter.extract_final_text(full_resp)
    ans, fmt = extract_answer(final_text)
    return parse_answer(ans, fmt)


async def _async_eval_sample(adapter, sampling_client, task_name, sample, ecfg):
    """Run n_samples inferences for one sample, return (label, prediction)."""
    label = int(sample["Y"])
    preds = []
    for _ in range(ecfg.n_samples):
        try:
            p = await _async_inference_single(
                adapter, sampling_client, task_name, sample["text"], ecfg
            )
            if p is not None:
                preds.append(p)
        except Exception as e:
            logger.warning(f"Eval inference exception: {e}")
    if not preds:
        return label, 1 - label, True  # failed
    pred = 1 if preds.count(1) >= preds.count(0) else 0
    return label, pred, False


async def _async_eval_task(adapter, sampling_client, task_name, data, ecfg):
    """Evaluate a single task with all samples in parallel, retrying failed parses up to 4 times."""
    max_retries = ecfg.eval_max_retries
    
    # Track state for each sample index: (label, preds_list, failed_boolean)
    results_state = {i: (int(s["Y"]), [], True) for i, s in enumerate(data)}
    
    for attempt_idx in range(max_retries):
        # Identify which samples still need to be evaluated (either failed or not started)
        to_run_indices = [i for i, state in results_state.items() if state[2]]
        
        if not to_run_indices:
            break  # All samples succeeded!
            
        logger.info(f"[{task_name}] Attempt {attempt_idx+1}/{max_retries} - Running {len(to_run_indices)} samples")
        
        coros = [
            _async_eval_sample(adapter, sampling_client, task_name, data[i], ecfg)
            for i in to_run_indices
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)
        
        for i, r in zip(to_run_indices, results):
            if isinstance(r, Exception):
                logger.warning(f"Eval sample exception in {task_name} (sample {i}): {r}")
                continue
                
            label, pred, failed = r
            
            # Update the state. If it stopped failing, we save the valid prediction.
            if not failed:
                results_state[i] = (label, [pred], False)

    # Aggregate final results
    y_true, y_pred, fails = [], [], 0
    for i, (label, preds, failed) in results_state.items():
        y_true.append(label)
        if failed:
            fails += 1
            y_pred.append(1 - label) # Force wrong prediction for F1 punishment
        else:
            y_pred.append(preds[0]) # Since we only store successful ones, preds[0] is the final pred

    f1 = f1_score(y_true, y_pred, average="macro", labels=[0,1]) if len(set(y_true)) > 1 else 0.0
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    if ecfg.verbose:
        logger.info(f"  {task_name}: F1={f1:.4f}, Acc={correct}/{len(data)}, Failed={fails}")
    return {"f1": f1, "n": len(data), "correct": correct, "failed": fails,
            "group": TASK_TO_GROUP.get(task_name, "Other")}

def run_eval(adapter, sampling_client, ecfg):
    logger.info("=" * 60 + "\nTDC EVALUATION\n" + "=" * 60)
    test_tasks = load_test_data(ecfg)
    if not test_tasks: return {"eval/macro_f1": 0.0}

    async def _all_tasks():
        gathered = await asyncio.gather(
            *[_async_eval_task(adapter, sampling_client, tn, td, ecfg)
              for tn, td in test_tasks.items()],
            return_exceptions=True,
        )
        task_results = {}
        for (tn, _), res in zip(test_tasks.items(), gathered):
            if isinstance(res, Exception):
                logger.warning(f"Task {tn} eval failed: {res}")
                task_results[tn] = {"f1": 0.0, "n": 0, "correct": 0, "failed": 0,
                                    "group": TASK_TO_GROUP.get(tn, "Other")}
            else:
                task_results[tn] = res
        return task_results

    results = asyncio.run(_all_tasks())

    per_group = {}
    for tn, r in results.items():
        per_group.setdefault(r["group"], []).append(r["f1"])
    group_means = {g: float(np.mean(fs)) for g, fs in per_group.items()}
    macro_f1 = float(np.mean([r["f1"] for r in results.values()]))

    for g in ["ADME","Tox","HTS","Other"]:
        if g in group_means:
            logger.info(f"  {g}: {group_means[g]:.4f}")
    logger.info(f"  MACRO F1: {macro_f1:.4f}")

    flat = {"eval/macro_f1": macro_f1}
    for tn, r in results.items(): flat[f"eval/{tn}/f1"] = r["f1"]
    for g, m in group_means.items(): flat[f"eval/{g}/mean_f1"] = m

    eval_metadata = getattr(ecfg, "eval_metadata", None) or {}
    flat.update(eval_metadata)

    if ecfg.log_dir:
        os.makedirs(ecfg.log_dir, exist_ok=True)
        with open(os.path.join(ecfg.log_dir, "eval_results.json"), "w") as f:
            json.dump({"macro_f1": macro_f1, "per_task": {t: r["f1"] for t,r in results.items()},
                        "per_group": group_means, "metadata": eval_metadata}, f, indent=2)
    return flat


def inline_eval(adapter, training_client, service_client, step, ecfg):
    path = training_client.save_weights_for_sampler(name=f"eval_{step:06d}").result().path
    sc = service_client.create_sampling_client(model_path=path)
    if ecfg.log_dir: ecfg.log_dir = os.path.join(ecfg.log_dir, f"eval_{step:06d}")
    return run_eval(adapter, sc, ecfg)
