"""Framework-independent audit workflow and input contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import pandas as pd
import numpy as np

from .metrics import comparison_table

REQUIRED_COLUMNS = {"text", "qualified", "decision"}
BINARY_COLUMNS = ("qualified", "decision", "group_b")


def validate_audit_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return a normalized copy and non-blocking warnings; raise on contract errors."""
    if frame.empty:
        raise ValueError("The dataset is empty.")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}.")

    result = frame.copy()
    result["text"] = result["text"].fillna("").astype(str).str.strip()
    if (result["text"] == "").any():
        count = int((result["text"] == "").sum())
        raise ValueError(f"The text column contains {count} blank record(s).")

    for column in BINARY_COLUMNS:
        if column not in result:
            continue
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must contain numeric 0/1 values.") from exc
        if not result[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 or 1.")
        result[column] = result[column].astype(int)

    warnings: list[str] = []
    if len(result) < 100:
        warnings.append("Fewer than 100 records: fairness estimates may be unstable.")
    if "group_b" not in result:
        warnings.append(
            "No benchmark group_b column was supplied; results are screening estimates and their error cannot be measured."
        )
    return result, warnings


def score_audit_frame(
    frame: pd.DataFrame,
    probability_predictor: Callable[[list[str]], object],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    normalized, warnings = validate_audit_frame(frame)
    result = normalized.copy()
    probabilities = np.asarray(probability_predictor(result["text"].tolist()), dtype=float)
    if probabilities.shape != (len(result),) or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Imputation must return one finite probability between 0 and 1 per record.")
    result["group_b_probability"] = probabilities
    result["imputed_group"] = result["group_b_probability"].map(
        lambda probability: "Group B" if probability >= 0.5 else "Group A"
    )
    return result, comparison_table(result), warnings


def build_audit_report(
    scored: pd.DataFrame,
    comparison: pd.DataFrame,
    model_source: str,
) -> dict:
    """Build a portable, non-identifying run summary."""
    return {
        "application": "FairnessLens",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_source": model_source,
        "record_count": int(len(scored)),
        "benchmark_available": "group_b" in scored.columns,
        "method": "soft demographic imputation",
        "threshold_for_display_only": 0.5,
        "fairness_metrics": comparison.to_dict(orient="index"),
        "disclaimer": (
            "Educational aggregate screening only; not an identity determination or legal compliance conclusion."
        ),
    }
