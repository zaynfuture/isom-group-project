"""Fine-tune a Hugging Face sequence classifier on synthetic proxy text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
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
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--sample-limit", type=int, default=None)
    return parser.parse_args()


def compute_metrics(evaluation):
    logits, labels = evaluation
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
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
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    (MODEL_DIR / "training_metadata.json").write_text(
        json.dumps(
            {
                "base_model": args.model_name,
                "tokenizer": tokenizer_name,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "seed": SEED,
                "test_metrics": test_metrics,
            },
            indent=2,
        )
    )
    print(json.dumps(test_metrics, indent=2))
    print(f"Saved fine-tuned model to {MODEL_DIR}")


if __name__ == "__main__":
    main()
