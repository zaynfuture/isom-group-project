"""Lazy Hugging Face inference helpers used by the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def model_is_available(model_dir: Path) -> bool:
    return (model_dir / "config.json").exists() and any(
        (model_dir / name).exists()
        for name in ("model.safetensors", "pytorch_model.bin")
    )


def load_pipeline(model_dir: Path):
    if not model_is_available(model_dir):
        raise FileNotFoundError(
            f"No fine-tuned model found at {model_dir}. Run: python scripts/train.py"
        )
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model=str(model_dir),
        tokenizer=str(model_dir),
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

