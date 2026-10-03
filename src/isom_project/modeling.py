"""Lazy Hugging Face inference helpers used by the Streamlit app."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("USE_TF", "0")


def model_is_available(model_dir: Path) -> bool:
    return (model_dir / "config.json").exists() and any(
        (model_dir / name).exists()
        for name in ("model.safetensors", "pytorch_model.bin")
    )


def load_pipeline(model_source: str | Path):
    model_source = str(model_source)
    local_path = Path(model_source)
    is_local = local_path.exists() or model_source.startswith(("/", "."))
    if is_local and not model_is_available(local_path):
        raise FileNotFoundError(
            f"No fine-tuned model found at {local_path}. Run: python scripts/train.py"
        )
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model=model_source,
        tokenizer=model_source,
        top_k=None,
        device=-1,
    )


def predict_probabilities(classifier, texts: list[str], batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.asarray([], dtype=float)
    outputs = classifier(texts, batch_size=batch_size, truncation=True, max_length=96)
    probabilities = []
    for row in outputs:
        mapping = {item["label"].upper(): float(item["score"]) for item in row}
        probability = mapping.get("GROUP_B", mapping.get("LABEL_1"))
        if probability is None:
            raise ValueError(f"Model output has no Group B label: {mapping}")
        probabilities.append(probability)
    return np.asarray(probabilities)
