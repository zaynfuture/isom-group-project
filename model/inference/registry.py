"""Two task-specific Transformer interfaces; no random-head fallback."""
import os
import json
from pathlib import Path
import numpy as np
os.environ.setdefault("USE_TF", "0")
BASE_MODEL = "distilbert/distilbert-base-uncased"
BASE_MODELS = {"merchant":"microsoft/MiniLM-L12-H384-uncased", "forecast":"amazon/chronos-bolt-tiny"}
TASKS = {"merchant": ["Department stores", "Dining", "Fuel", "Grocery", "Lodging"], "forecast": ["HIGH", "LOW", "MEDIUM"]}
ROOT = Path(__file__).resolve().parents[2]


def deployed(task):
    path = ROOT / "model" / "artifacts" / "spendlens_deployment.json"
    return json.loads(path.read_text()).get(task, {}) if path.exists() else {}


def source_for(task):
    return os.getenv(f"SPENDLENS_{task.upper()}_MODEL", deployed(task).get("repo_id", str(ROOT / "model" / "artifacts" / f"spendlens_{task}")))


def available(task):
    return bool(os.getenv(f"SPENDLENS_{task.upper()}_MODEL")) or bool(deployed(task)) or (Path(source_for(task)) / "model.safetensors").exists()


def load(task):
    if not available(task):
        raise ValueError("Awaiting fine-tuning and model upload.")
    from transformers import pipeline
    import torch
    torch.set_num_threads(2)
    revision = None if os.getenv(f"SPENDLENS_{task.upper()}_MODEL") else deployed(task).get("revision")
    if task == "forecast" and deployed(task).get("architecture") == "chronos-bolt":
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained(source_for(task),revision=revision,device_map="cpu",torch_dtype=torch.float32)
        if pipe.__class__.__name__ != "ChronosBoltPipeline":
            raise ValueError("Forecast checkpoint must implement the Chronos-Bolt contract.")
        return pipe
    pipe = pipeline("text-classification", model=source_for(task), tokenizer=source_for(task), revision=revision, device=-1, top_k=None, framework="pt")
    if set(pipe.model.config.id2label.values()) != set(TASKS[task]):
        raise ValueError("Checkpoint label schema does not match this task.")
    return pipe


def infer(pipe, texts):
    outputs = pipe(texts, truncation=True, max_length=256, batch_size=16)
    best = [max(row, key=lambda item: item["score"]) for row in outputs]
    return [row["label"] for row in best], np.array([row["score"] for row in best])
