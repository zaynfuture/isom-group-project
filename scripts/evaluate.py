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
    classifier = load_pipeline(MODEL_DIR)
    split_results = {}
    scored_splits = {}
    for split in ("train", "validation", "test"):
        part = frame.loc[frame.split == split].copy()
        part["group_b_probability"] = predict_probabilities(classifier, part.text.tolist())
        scored_splits[split] = part
        split_results[split] = {
            "records": len(part),
            **classification_metrics(part.group_b, part.group_b_probability),
        }
    test = scored_splits["test"]

    classification = classification_metrics(test.group_b, test.group_b_probability)
    fairness = comparison_table(test).to_dict(orient="index")
    output = {"classification": classification, "fairness": fairness,
              "split_evaluation": split_results,
              "evaluation_protocol": "Post-training evaluation of the saved checkpoint. Training scores are in-sample; validation selected the checkpoint; test scores are held-out estimates. Test results have been inspected across project iterations; this is not a one-shot external validation."}

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(output, indent=2, allow_nan=False))
    test.to_csv(ARTIFACT_DIR / "evaluation_predictions.csv", index=False)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
