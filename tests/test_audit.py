from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import pytest

from isom_project.audit import build_audit_report, score_audit_frame, validate_audit_frame


def valid_frame(include_group=True):
    data = {
        "text": [f"synthetic profile {i}" for i in range(120)],
        "qualified": [i % 2 for i in range(120)],
        "decision": [(i // 2) % 2 for i in range(120)],
    }
    if include_group:
        data["group_b"] = [i % 2 for i in range(120)]
    return pd.DataFrame(data)


def test_missing_column_is_rejected():
    with pytest.raises(ValueError, match="Missing required columns: decision"):
        validate_audit_frame(valid_frame().drop(columns="decision"))


def test_invalid_binary_value_is_rejected():
    frame = valid_frame()
    frame.loc[0, "qualified"] = 3
    with pytest.raises(ValueError, match="qualified must contain only 0 or 1"):
        validate_audit_frame(frame)


def test_scoring_and_report_contract():
    scored, comparison, warnings = score_audit_frame(
        valid_frame(), lambda texts: np.linspace(0.05, 0.95, len(texts))
    )
    report = build_audit_report(scored, comparison, "test-model")
    assert not warnings
    assert "group_b_probability" in scored
    assert list(comparison.index) == ["Observed group", "Imputed (soft)", "Absolute error"]
    assert report["record_count"] == 120
    assert report["benchmark_available"] is True
