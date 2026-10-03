"""Evaluate classification and downstream fairness preservation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from isom_project.config import ARTIFACT_DIR, DATA_DIR, METRICS_PATH, MODEL_DIR
from isom_project.metrics import classification_metrics, comparison_table
from isom_project.modeling import load_pipeline, predict_probabilities


def main():
    frame = pd.read_csv(DATA_DIR / "synthetic_audit_data.csv")
    test = frame.loc[frame.split == "test"].copy()
    classifier = load_pipeline(MODEL_DIR)
    test["group_b_probability"] = predict_probabilities(classifier, test.text.tolist())

    classification = classification_metrics(test.group_b, test.group_b_probability)
    fairness = comparison_table(test).to_dict(orient="index")
    output = {"classification": classification, "fairness": fairness}

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(output, indent=2, allow_nan=False))
    test.to_csv(ARTIFACT_DIR / "evaluation_predictions.csv", index=False)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

