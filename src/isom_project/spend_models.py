"""Two task-specific Transformer interfaces; no random-head fallback."""
import os
from pathlib import Path
import numpy as np
os.environ.setdefault("USE_TF", "0")
BASE_MODEL = "distilbert/distilbert-base-uncased"
TASKS = {"merchant": ["Department stores", "Dining", "Fuel", "Grocery", "Lodging"], "forecast": ["HIGH", "LOW", "MEDIUM"]}
ROOT = Path(__file__).resolve().parents[2]


def source_for(task):
    return os.getenv(f"SPENDLENS_{task.upper()}_MODEL", str(ROOT / "artifacts" / f"spendlens_{task}"))


def available(task):
    return bool(os.getenv(f"SPENDLENS_{task.upper()}_MODEL")) or (Path(source_for(task)) / "model.safetensors").exists()


def load(task):
    if not available(task):
        raise ValueError("Awaiting fine-tuning and model upload.")
    from transformers import pipeline
    pipe = pipeline("text-classification", model=source_for(task), tokenizer=source_for(task), device=-1, top_k=None)
    if set(pipe.model.config.id2label.values()) != set(TASKS[task]):
        raise ValueError("Checkpoint label schema does not match this task.")
    return pipe


def infer(pipe, texts):
    outputs = pipe(texts, truncation=True, max_length=256, batch_size=16)
    best = [max(row, key=lambda item: item["score"]) for row in outputs]
    return [row["label"] for row in best], np.array([row["score"] for row in best])
