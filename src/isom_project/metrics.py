"""Classification and downstream fairness metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, probability) -> dict[str, float | list[list[int]]]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def _weighted_rate(values, weights, mask=None) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        values, weights = values[mask], weights[mask]
    if len(values) == 0 or weights.sum() <= 0:
        return float("nan")
    return float(np.average(values, weights=weights))


def fairness_metrics(decision, qualified, group_probability) -> dict[str, float]:
    """Compute soft group-B minus group-A fairness metrics."""
    d = np.asarray(decision, dtype=int)
    q = np.asarray(qualified, dtype=int)
    p = np.clip(np.asarray(group_probability, dtype=float), 0.0, 1.0)
    rate_b = _weighted_rate(d, p)
    rate_a = _weighted_rate(d, 1 - p)
    tpr_b = _weighted_rate(d, p, q == 1)
    tpr_a = _weighted_rate(d, 1 - p, q == 1)
    fpr_b = _weighted_rate(d, p, q == 0)
    fpr_a = _weighted_rate(d, 1 - p, q == 0)
    return {
        "group_a_selection_rate": rate_a,
        "group_b_selection_rate": rate_b,
        "demographic_parity_gap": rate_b - rate_a,
        "disparate_impact_ratio": rate_b / rate_a if rate_a > 0 else float("nan"),
        "equal_opportunity_gap": tpr_b - tpr_a,
        "false_positive_rate_gap": fpr_b - fpr_a,
    }


def comparison_table(frame: pd.DataFrame, probability_column: str = "group_b_probability") -> pd.DataFrame:
    predicted = fairness_metrics(frame.decision, frame.qualified, frame[probability_column])
    rows = {"Imputed (soft)": predicted}
    if "group_b" in frame.columns:
        observed = fairness_metrics(frame.decision, frame.qualified, frame.group_b)
        rows = {"Observed group": observed, **rows}
    result = pd.DataFrame(rows).T
    if "Observed group" in result.index:
        result.loc["Absolute error"] = (
            result.loc["Imputed (soft)"] - result.loc["Observed group"]
        ).abs()
    return result

