"""
TDC Reward Function for Slime RL Training

Custom reward function that evaluates whether the model's final answer
matches the ground truth label for TDC binary classification tasks.

Usage in shell script:
    --custom-rm-path reward_tdc.reward_func
"""

import logging
import os
import sys
import numpy as np

# Add project root to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils.TDC_answer_parser import extract_answer, parse_answer

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """
    Compute reward for a TDC binary classification sample.

    Reward scheme:
    - 1.0 for correct prediction
    - 0.0 for incorrect prediction or unparseable answer
    - +0.1 bonus for correct "Answer:" format (encourages format compliance)

    Args:
        args: Rollout arguments from Slime training pipeline
        sample: Sample containing the model response and ground truth label
        **kwargs: Additional arguments

    Returns:
        Float reward value
    """
    response_text = sample.final_content or ""

    # Get ground truth label (Y field: 0 or 1)
    ground_truth = None
    if sample.label is not None:
        try:
            ground_truth = int(sample.label)
        except (ValueError, TypeError):
            pass

    if ground_truth is None:
        # Try metadata
        metadata = sample.metadata or {}
        try:
            ground_truth = int(metadata.get("Y", metadata.get("label", -1)))
            if ground_truth not in (0, 1):
                # Try parsing label string like "(A)" or "(B)"
                label_str = str(metadata.get("label", ""))
                if "(B)" in label_str:
                    ground_truth = 1
                elif "(A)" in label_str:
                    ground_truth = 0
                else:
                    ground_truth = None
        except (ValueError, TypeError):
            ground_truth = None

    if ground_truth is None:
        logger.warning(f"Could not determine ground truth for sample index={sample.index}")
        return 0.0

    # Extract and parse the model's answer
    answer_text, format_correct = extract_answer(response_text)
    prediction = parse_answer(answer_text, format_correct, think_is_on=True)

    if prediction is not None and prediction == ground_truth:
        ans_reward = 1.0
    elif prediction is not None:
        ans_reward = 0.0
    else:
        # Could not parse answer
        ans_reward = 0.0

    # Format bonus: encourage the model to use "Answer:" prefix
    if format_correct:
        format_reward = 0.05
    else:
        format_reward = 0.0

    tool_reward = 0.0
    if sample.metadata is not None:
        num_tool_calls = sample.metadata.get("num_tool_calls", 0)
        tool_reward = 0.07 * np.tanh(1 * num_tool_calls)

    response_length = getattr(sample, 'response_length', 0)
    if response_length <= 4000:
        length_penalty = 0.0
    elif response_length >= 6000:
        length_penalty = -1.0
    else:
        length_penalty = - (response_length - 4000) / 2000.0

    total_reward = ans_reward + format_reward + ans_reward * tool_reward + length_penalty
    
    # Store individual components in metadata for wandb logging
    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["ans_reward"] = ans_reward
    sample.metadata["format_reward"] = format_reward
    sample.metadata["tool_reward"] = ans_reward * tool_reward
    sample.metadata["length_penalty"] = length_penalty
    sample.metadata["total_reward"] = total_reward

    logger.debug(
        f"Reward: {total_reward}, prediction={prediction}, ground_truth={ground_truth}, "
        f"format_ok={format_correct}, sample_index={sample.index}"
    )

    return total_reward
