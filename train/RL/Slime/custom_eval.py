import numpy as np
import logging
from sklearn.metrics import f1_score
from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step
# Import the TDC Answer Parser as per original user test logic
from utils.TDC_answer_parser import extract_answer, parse_answer

logger = logging.getLogger(__name__)

def log_eval_rollout_data_f1(rollout_id, args, data: dict, extra_metrics: dict) -> bool:
    """
    Slime custom Eval log function.
    Returns True to indicate we takeover the default logging behavior.
    Calculates the Macro-F1 score based on majority voting.
    """
    log_dict = extra_metrics or {}
    
    all_f1_scores = []
    all_rewards = []
    
    # Process each dataset individually (e.g. BBB_Martins_test)
    for dataset_name, dataset_info in data.items():
        samples = dataset_info["samples"]
        
        y_true = []
        y_pred = []
        
        # Group samples by the original prompt/question index
        group_to_samples = {}
        for sample in samples:
            group_to_samples.setdefault(sample.group_index, []).append(sample)
            
        for g_idx, group_samples in group_to_samples.items():
            # Get the ground truth label.
            # The JSONL has "Y" at the top level and "label": "(B)" inside metadata.
            # Slime passes --metadata-key metadata which populates Sample.metadata.
            truth = None
            if hasattr(group_samples[0], "metadata") and group_samples[0].metadata is not None:
                label_str = group_samples[0].metadata.get("label", None)
                if label_str == "(A)":
                    truth = 0
                elif label_str == "(B)":
                    truth = 1
            
            if truth is None:
                logger.warning(f"Could not find ground truth 'label' in metadata for eval group {g_idx}. Metadata: {group_samples[0].metadata}")
                continue
                
            y_true.append(truth)
            
            valid_preds = []
            for sample in group_samples:
                ans_txt, fmt_ok = extract_answer(sample.response)
                # The assumption is thinking logic was enabled for generation during eval.
                # If evaluating without thinking, change think_is_on to False or pull from args conditionally.
                # Since training script has args, we can approximate checking if think is enabled (e.g. args.sglang_speculative_algorithm or whatever logic).
                pred = parse_answer(ans_txt, format_correct=fmt_ok, think_is_on=True)
                
                if pred is not None:
                    valid_preds.append(pred)
            
            # Majority voting
            if not valid_preds:
                # If everything failed to parse, punish by predicting the opposite
                y_pred.append(1 - truth) 
            else:
                count_1 = valid_preds.count(1)
                count_0 = valid_preds.count(0)
                y_pred.append(1 if count_1 >= count_0 else 0)
        
        score = 0.0
        if len(set(y_true)) > 1:
            score = f1_score(y_true, y_pred, average='macro', pos_label=1)
        else:
            # Matching test_tdc_via_api_F1.py log logic for single class
            pass
            
        all_f1_scores.append(score)
            
        # Add the F1 score to the logs
        log_dict[f"eval/F1_score/{dataset_name}"] = score
        
        # We can also keep the pass rate or rewards if they were calculated
        rewards = dataset_info.get("rewards", [])
        if rewards:
            task_reward = sum(rewards) / len(rewards)
            log_dict[f"eval/reward/{dataset_name}"] = task_reward
            all_rewards.append(task_reward)
            
        logger.info(f"Eval F1 Score for {dataset_name}: {score:.4f} (based on {len(y_true)} samples)")
        
    if all_f1_scores:
        macro_avg_f1 = sum(all_f1_scores) / len(all_f1_scores)
        log_dict["eval/F1_score/macro_average"] = macro_avg_f1
        logger.info(f"Cross-task Macro-Average F1: {macro_avg_f1:.4f} (over {len(all_f1_scores)} tasks)")
        
    if all_rewards:
        macro_avg_reward = sum(all_rewards) / len(all_rewards)
        log_dict["eval/reward/macro_average"] = macro_avg_reward
        logger.info(f"Cross-task Macro-Average Reward: {macro_avg_reward:.4f} (over {len(all_rewards)} tasks)")
        
    # Standard Slime metric reporting
    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    
    # [BUG FIX]: Align the Global WandB "Step" (X-axis) with the semantic "eval/step"
    # Using `wandb.define_metric` forces WandB to use our custom `eval/step` as the 
    # X-axis for all metrics with the "eval/" prefix, instead of its own global "Step".
    import wandb
    if getattr(args, "use_wandb", False) and wandb.run is not None:
        # Define eval/step as the default x-axis for all eval metrics
        wandb.define_metric("eval/step")
        wandb.define_metric("eval/*", step_metric="eval/step")
        wandb.log(log_dict)
    else:
        logging_utils.log(args, log_dict, step_key="eval/step")

    # True indicates Slime's default evaluator doesn't need to redundantly print its generic logs again.
    return True
