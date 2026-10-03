"""Fine-tune a Hugging Face sequence classifier on synthetic proxy text."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

# This project is PyTorch-only. Prevent Transformers 4 from importing an
# incompatible TensorFlow/Keras stack when one happens to be preinstalled.
os.environ.setdefault("USE_TF", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from isom_project.config import ARTIFACT_DIR, DATA_DIR, MODEL_DIR, SEED
from isom_project.data import write_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="google/bert_uncased_L-2_H-128_A-2")
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def compute_metrics(evaluation):
    logits, labels = evaluation
    predictions = np.argmax(logits, axis=-1)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "roc_auc": roc_auc_score(labels, probabilities[:, 1]),
        "brier_score": brier_score_loss(labels, probabilities[:, 1]),
    }


def main():
    args = parse_args()
    set_seed(SEED)
    data_path = DATA_DIR / "synthetic_audit_data.csv"
    if not data_path.exists():
        write_dataset()
    frame = pd.read_csv(data_path)
    if args.sample_limit:
        frame = frame.groupby("split", group_keys=False).head(args.sample_limit)

    tokenizer_name = args.tokenizer_name or (
        "bert-base-uncased"
        if args.model_name in {"prajjwal1/bert-tiny", "google/bert_uncased_L-2_H-128_A-2"}
        else args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data = DatasetDict(
        {
            split: Dataset.from_pandas(
                part[["text", "group_b"]].rename(columns={"group_b": "label"}),
                preserve_index=False,
            )
            for split, part in frame.groupby("split")
        }
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized = data.map(tokenize, batched=True, remove_columns=["text"])
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "GROUP_A", 1: "GROUP_B"},
        label2id={"GROUP_A": 0, "GROUP_B": 1},
    )
    output_dir = ARTIFACT_DIR / "trainer"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=25,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    trainer.train()
    # Early stopping only applies to validation checkpoints. Removing the
    # callback avoids a misleading missing `eval_macro_f1` warning when the
    # held-out split is evaluated with the `test_` metric prefix below.
    trainer.remove_callback(EarlyStoppingCallback)
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    # Course-aligned save/reload test: serialized weights must reproduce logits.
    verification_texts = [
        "profile token alden; region north; channel web; product credit; income band 3; tenure 5 years",
        "profile token pavan; region south; channel branch; product savings; income band 4; tenure 8 years",
    ]
    encoded = tokenizer(
        verification_texts, padding=True, truncation=True,
        max_length=args.max_length, return_tensors="pt"
    )
    model_device = next(trainer.model.parameters()).device
    encoded = {name: value.to(model_device) for name, value in encoded.items()}
    trainer.model.eval()
    with torch.no_grad():
        before_save = trainer.model(**encoded).logits.detach().cpu().numpy()

    reloaded = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(model_device)
    reloaded.eval()
    encoded_reloaded = tokenizer(
        verification_texts, padding=True, truncation=True,
        max_length=args.max_length, return_tensors="pt"
    )
    encoded_reloaded = {
        name: value.to(model_device) for name, value in encoded_reloaded.items()
    }
    with torch.no_grad():
        after_reload = reloaded(**encoded_reloaded).logits.detach().cpu().numpy()
    reload_max_absolute_difference = float(np.max(np.abs(before_save - after_reload)))
    if not np.allclose(before_save, after_reload, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Saved and reloaded model predictions do not match")

    (MODEL_DIR / "training_metadata.json").write_text(
        json.dumps(
            {
                "base_model": args.model_name,
                "tokenizer": tokenizer_name,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "max_length": args.max_length,
                "early_stopping_patience": args.early_stopping_patience,
                "seed": SEED,
                "test_metrics": test_metrics,
                "reload_max_absolute_difference": reload_max_absolute_difference,
            },
            indent=2,
        )
    )

    model_card = f"""---
pipeline_tag: text-classification
library_name: transformers
base_model: {args.model_name}
---

# FairnessLens synthetic demographic imputer

This classroom model was fine-tuned on synthetic Group A/Group B proxy text for
testing downstream fairness-estimation error. It is not validated for real
demographic inference and must not be used for individual decisions.

## Held-out test metrics

```json
{json.dumps(test_metrics, indent=2)}
```
"""
    (MODEL_DIR / "README.md").write_text(model_card)
    (MODEL_DIR / "training_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2)
    )

    archive_path = shutil.make_archive(
        str(ARTIFACT_DIR / "fairnesslens_model"),
        "zip",
        root_dir=MODEL_DIR.parent,
        base_dir=MODEL_DIR.name,
    )

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--hub-model-id is required with --push-to-hub")
        reloaded.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
        print(f"Uploaded model and tokenizer to https://huggingface.co/{args.hub_model_id}")

    print(json.dumps(test_metrics, indent=2))
    print(f"Saved fine-tuned model to {MODEL_DIR}")
    print(f"Reload verification max difference: {reload_max_absolute_difference:.2e}")
    print(f"Packaged model: {archive_path}")


if __name__ == "__main__":
    main()
