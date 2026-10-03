from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from isom_project.metrics import classification_metrics, comparison_table, fairness_metrics


def test_perfect_classification_metrics():
    result = classification_metrics([0, 0, 1, 1], [0.01, 0.20, 0.80, 0.99])
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["roc_auc"] == 1.0


def test_fairness_gap_direction():
    result = fairness_metrics(
        decision=[1, 1, 0, 0],
        qualified=[1, 1, 1, 1],
        group_probability=[0, 0, 1, 1],
    )
    assert result["demographic_parity_gap"] == -1.0
    assert result["equal_opportunity_gap"] == -1.0


def test_comparison_has_error_row():
    frame = pd.DataFrame(
        {
            "decision": [1, 0, 1, 0],
            "qualified": [1, 1, 0, 0],
            "group_b": [0, 0, 1, 1],
            "group_b_probability": [0.1, 0.2, 0.8, 0.9],
        }
    )
    result = comparison_table(frame)
    assert list(result.index) == ["Observed group", "Imputed (soft)", "Absolute error"]
    assert np.isfinite(result.loc["Absolute error", "demographic_parity_gap"])

