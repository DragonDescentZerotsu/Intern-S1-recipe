import re
from utils.TDC_answer_parser import extract_answer, parse_answer
from config import cfg


def compute_reward(full_text: str, final_text: str, gt: int) -> float:
    """Exact copy of reward_tdc.reward_func reward logic."""
    answer_text, format_correct = extract_answer(final_text)
    prediction = parse_answer(answer_text, format_correct)
    reward = 0.0
    if prediction is not None and prediction == gt:
        reward = 1.0
        # tool use reward
        # if cfg.use_tools and ("to=functions." in full_text or "TOOL CALL" in full_text):
        #     reward += 0.1
    # “Answer: (A) or Answer: (B)” in final text
    if format_correct:
        reward += cfg.format_bonus
    return reward
