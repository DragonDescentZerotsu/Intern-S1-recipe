from __future__ import annotations

import math

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def safe_macro_f1(y_true, y_pred) -> float:
    if len(set(y_true)) <= 1:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro"))


def safe_roc_auc(y_true, y_score) -> float:
    if len(set(y_true)) <= 1:
        return math.nan
    return float(roc_auc_score(y_true, y_score))


def compute_binary_classification_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": safe_macro_f1(y_true, y_pred),
        "roc_auc": safe_roc_auc(y_true, y_score),
    }

