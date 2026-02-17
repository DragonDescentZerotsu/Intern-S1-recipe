"""
TDC Reward Function for Slime RL Training

Custom reward function that evaluates whether the model's final answer
matches the ground truth label for TDC binary classification tasks.

Usage in shell script:
    --custom-rm-path reward_tdc.reward_func
"""

import logging
import re

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def extract_answer(response: str):
    """
    Extract the final answer from model response.
    Looks for 'Answer:', 'answer is', or '<answer>' patterns.
    Returns (answer_text, format_correct).
    """
    format_correct = False
    answer_matches = None

    if "Answer:" in response:
        format_correct = True
        answer_matches = list(re.finditer(r"Answer:", response, re.IGNORECASE))
    elif "answer is" in response:
        format_correct = True
        answer_matches = list(re.finditer(r"answer is", response, re.IGNORECASE))

    if answer_matches:
        last_answer_pos = answer_matches[-1].end()
        answer_text = response[last_answer_pos:].strip()
    else:
        answer_text = response

    return answer_text, format_correct


def parse_answer(answer_text, format_correct):
    """
    Parse answer text to binary prediction: (A) -> 0 (negative), (B) -> 1 (positive).
    """
    if answer_text is None:
        return None

    if format_correct:
        if "(A)" in answer_text:
            return 0
        elif "A**" in answer_text:
            return 0
        elif "A)" in answer_text:
            return 0
        elif "\\boxed{A}" in answer_text:
            return 0
        elif "\\text{A}" in answer_text:
            return 0
        elif "(B)" in answer_text:
            return 1
        elif "B**" in answer_text:
            return 1
        elif "B)" in answer_text:
            return 1
        elif "\\boxed{B}" in answer_text:
            return 1
        elif "\\text{B}" in answer_text:
            return 1
        elif "B" in answer_text:
            return 1
        elif "A" in answer_text:
            return 0
        else:
            return None
    else:
        if "(A)" in answer_text:
            return 0
        elif "(B)" in answer_text:
            return 1
        elif "Yes" in answer_text:
            return 1
        elif "yes" in answer_text:
            return 1
        elif "B" in answer_text:
            return 1
        elif "A" in answer_text:
            return 0
        else:
            return None


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
    response_text = sample.response or ""

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
    prediction = parse_answer(answer_text, format_correct)

    # Compute reward
    reward = 0.0

    if prediction is not None and prediction == ground_truth:
        reward = 1.0
    elif prediction is not None:
        reward = 0.0
    else:
        # Could not parse answer
        reward = 0.0

    # Format bonus: encourage the model to use "Answer:" prefix
    if format_correct:
        reward += 0.1

    logger.debug(
        f"Reward: {reward}, prediction={prediction}, ground_truth={ground_truth}, "
        f"format_ok={format_correct}, sample_index={sample.index}"
    )

    return reward
