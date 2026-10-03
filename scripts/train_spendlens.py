"""Explicit offline fine-tuning for both SpendLens tasks. Never runs on app startup."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault("USE_TF", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed
from isom_project.spending import demo_transactions, validate_transactions, monthly_examples
from isom_project.spend_models import BASE_MODEL, TASKS


def prepare(task, frame):
    if task == "merchant":
        examples = frame.loc[frame.category.isin(TASKS[task])].copy()
        examples["text"] = examples.merchant_description
        examples["label"] = examples.category
        cards = np.array(sorted(examples.card_id.unique()))
        np.random.default_rng(5240).shuffle(cards)
        n = len(cards)
        if n < 10:
            raise ValueError("Merchant training requires at least 10 cards.")
        train_cards, valid_cards = set(cards[:int(n*.7)]), set(cards[int(n*.7):int(n*.85)])
        examples["split"] = examples.card_id.map(lambda c: "train" if c in train_cards else "validation" if c in valid_cards else "test")
    else:
        if set(frame.currency) != {"USD"}:
            raise ValueError("Forecast demo thresholds require USD only.")
        examples = monthly_examples(frame)
        if examples.empty or examples.target_month.nunique() < 3:
            raise ValueError("Forecast training requires at least six calendar months of complete coverage.")
        months = sorted(examples.target_month.unique())
        examples["split"] = examples.target_month.map(lambda m: "test" if m == months[-1] else "validation" if m == months[-2] else "train")
    for split in ("train", "validation", "test"):
        if examples.loc[examples.split.eq(split)].empty:
            raise ValueError(f"Empty {split} split.")
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASKS))
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--model-name", default=BASE_MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    set_seed(5240)
    frame = validate_transactions(pd.read_csv(args.csv, dtype={"mcc":str,"card_id":str})) if args.csv else demo_transactions()
    examples = prepare(args.task, frame)
    labels = TASKS[args.task]
    mapping = {label:i for i,label in enumerate(labels)}
    from huggingface_hub import HfApi
    revision = HfApi().model_info(args.model_name, revision=args.revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, revision=revision, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=mapping)
    sets = {}
    for split in ("train", "validation", "test"):
        part = examples.loc[examples.split.eq(split), ["text","label"]].copy()
        part["label"] = part.label.map(mapping)
        sets[split] = Dataset.from_pandas(part, preserve_index=False).map(lambda batch:tokenizer(batch["text"], truncation=True, max_length=256), batched=True, remove_columns=["text"])
    def metrics(output):
        pred = output.predictions.argmax(-1)
        return {"accuracy":accuracy_score(output.label_ids,pred), "macro_f1":f1_score(output.label_ids,pred,labels=list(range(len(labels))),average="macro",zero_division=0)}
    output = args.output_dir or ROOT / "artifacts" / f"spendlens_{args.task}"
    if (output / "model.safetensors").exists():
        raise ValueError("Output already contains a model. Choose a new --output-dir to preserve prior runs.")
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(output / "checkpoints"),num_train_epochs=args.epochs,learning_rate=3e-5,per_device_train_batch_size=16,per_device_eval_batch_size=32,eval_strategy="epoch",save_strategy="epoch",load_best_model_at_end=True,metric_for_best_model="macro_f1",save_total_limit=1,report_to="none",seed=5240), train_dataset=sets["train"],eval_dataset=sets["validation"],processing_class=tokenizer,data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=metrics)
    started = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - started
    evaluation = {s:trainer.evaluate(sets[s],metric_key_prefix=s) for s in sets}
    test = examples.loc[examples.split.eq("test")]
    majority = examples.loc[examples.split.eq("train"),"label"].mode()[0]
    baseline = test.baseline_label if args.task == "forecast" else [majority]*len(test)
    evaluation["baseline"] = {"name":"previous two-month average" if args.task == "forecast" else "training majority class", "macro_f1":f1_score(test.label,baseline,labels=labels,average="macro",zero_division=0)}
    prediction = trainer.predict(sets["test"])
    evaluation["confusion_matrix"] = confusion_matrix(prediction.label_ids,prediction.predictions.argmax(-1),labels=list(range(len(labels)))).tolist()
    evaluation["labels"] = labels
    evaluation["split_counts"] = examples.split.value_counts().to_dict()
    evaluation["base_model"] = args.model_name
    evaluation["base_revision"] = revision
    evaluation["parameter_count"] = sum(p.numel() for p in model.parameters())
    evaluation["training_seconds"] = training_seconds
    evaluation["selection_metric"] = "validation macro_f1; test is not used for checkpoint selection"
    evaluation["best_checkpoint"] = Path(trainer.state.best_model_checkpoint).name
    evaluation["data_source"] = "user CSV" if args.csv else "synthetic demo"
    evaluation["limitations"] = "Requires representative external validation. Card splits can share merchants; forecasting assumes complete calendar-month coverage. Demo amounts are USD."
    if args.task == "merchant" and not args.csv:
        evaluation["limitations"] += " Five category-specific description templates recur across splits. Perfect template scores do not establish unseen-merchant generalization. Prior experiments inspected these test targets. Softmax scores are uncalibrated."
    trainer.save_model(output)
    tokenizer.save_pretrained(output)
    (output / "evaluation.json").write_text(json.dumps(evaluation,indent=2))
    (output / "training_history.json").write_text(json.dumps(trainer.state.log_history,indent=2))
    (output / "README.md").write_text(f"---\nlibrary_name: transformers\npipeline_tag: text-classification\nbase_model: {args.model_name}\n---\n# SpendLens {args.task}\n\nTask-specific fine-tuned model. Labels: {labels}.\n\nTraining data source: {evaluation['data_source']}. No demographic identity inference. Evaluation is in evaluation.json.\n\n{evaluation['limitations']}\n")
    card = (output / "README.md").read_text()
    if args.model_name == "microsoft/MiniLM-L12-H384-uncased":
        card = card.replace("library_name: transformers", "library_name: transformers\nlicense: mit")
    card += f"\nTest Macro-F1: {evaluation['test']['test_macro_f1']:.4f}; majority/baseline Macro-F1: {evaluation['baseline']['macro_f1']:.4f}. Parameter count: {evaluation['parameter_count']}.\n"
    (output / "README.md").write_text(card)
    reloaded = AutoModelForSequenceClassification.from_pretrained(output)
    import torch
    trainer.model.cpu().eval(); reloaded.eval()
    inputs = tokenizer(examples.text.iloc[:2].tolist(),padding=True,truncation=True,max_length=256,return_tensors="pt")
    with torch.no_grad():
        assert torch.allclose(trainer.model(**inputs).logits,reloaded(**inputs).logits,atol=1e-5)
    (output / "reload_verification.json").write_text(json.dumps({"passed":True,"atol":1e-5,"samples":2}))
    print(f"Saved and reload-verified: {output}. Upload separately after reviewing evaluation.json.")


if __name__ == "__main__":
    main()
