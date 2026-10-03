"""Fine-tune official Chronos-Bolt weights on numeric card spending histories."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from chronos import BaseChronosPipeline
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, mean_absolute_error
from transformers import set_seed
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from isom_project.spending import demo_transactions, validate_transactions
from isom_project.forecasting import BASE_FORECAST_MODEL, BANDS, numeric_examples, predict_amounts, spending_bands


def evaluate(pipe, part):
    result = predict_amounts(pipe, part.context.tolist())
    y = part.target.to_numpy()
    quantiles = result[["p10", "median_amount", "p90"]].to_numpy()
    error = y[:,None] - quantiles
    levels = np.array([.1,.5,.9])
    return {"mae":float(mean_absolute_error(y, result.median_amount)),
            "macro_f1":float(f1_score(part.label,result.prediction,labels=BANDS,average="macro",zero_division=0)),
            "accuracy":float(accuracy_score(part.label,result.prediction)),
            "mean_pinball_loss":float(np.maximum(levels*error,(levels-1)*error).mean()),
            "interval_80_coverage":float(np.mean((y >= result.p10) & (y <= result.p90))),
            "interval_80_mean_width":float((result.p90-result.p10).mean())}, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"artifacts/spendlens_forecast_chronos")
    args = parser.parse_args()
    out = args.output_dir
    if (out/"model.safetensors").exists():
        raise ValueError("Output already contains a model; use a new directory.")
    if args.epochs < 1:
        raise ValueError("At least one fine-tuning epoch is required.")
    set_seed(5240); torch.set_num_threads(2)
    frame = validate_transactions(pd.read_csv(args.csv,dtype={"card_id":str,"mcc":str})) if args.csv else demo_transactions()
    examples = numeric_examples(frame)
    if examples.empty or examples.target_month.nunique() < 3:
        raise ValueError("At least six months with complete account coverage are required.")
    months = sorted(examples.target_month.unique())
    parts = {"train":examples.loc[examples.target_month < months[-2]],
             "validation":examples.loc[examples.target_month == months[-2]],
             "test":examples.loc[examples.target_month == months[-1]]}
    revision = HfApi().model_info(BASE_FORECAST_MODEL).sha
    pipe = BaseChronosPipeline.from_pretrained(BASE_FORECAST_MODEL,revision=revision,device_map=args.device,torch_dtype=torch.float32)
    model = pipe.model
    pretrained_validation, _ = evaluate(pipe, parts["validation"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    train = parts["train"]
    context = torch.tensor(train.context.tolist(),dtype=torch.float32,device=args.device)
    target = torch.tensor(train.target.to_numpy(),dtype=torch.float32,device=args.device).unsqueeze(-1)
    best = float("inf"); history = []; best_epoch = None
    started = time.perf_counter()
    out.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,args.epochs+1):
        model.train(); losses = []
        for indices in torch.randperm(len(train)).split(32):
            optimizer.zero_grad()
            loss = model(context=context[indices],target=target[indices]).loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; no model will be published.")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val, _ = evaluate(pipe,parts["validation"])
        entry = {"epoch":epoch,"train_loss":float(np.mean(losses)),"eval_loss":val["mean_pinball_loss"],"eval_macro_f1":val["macro_f1"],"validation_mae":val["mae"]}
        history.append(entry); print(json.dumps(entry),flush=True)
        if val["mean_pinball_loss"] < best:
            best = val["mean_pinball_loss"]; best_epoch = epoch
            model.save_pretrained(out,safe_serialization=True)
    seconds = time.perf_counter() - started
    reloaded = BaseChronosPipeline.from_pretrained(str(out),device_map=args.device,torch_dtype=torch.float32)
    evaluation = {}
    for name, part in parts.items():
        metrics, predictions = evaluate(reloaded,part)
        evaluation[name] = {f"{name}_{key}":value for key,value in metrics.items()}
        if name == "test":
            evaluation["confusion_matrix"] = confusion_matrix(part.label,predictions.prediction,labels=BANDS).tolist()
    test = parts["test"]
    evaluation.update({"base_model":BASE_FORECAST_MODEL,"base_revision":revision,"architecture":"chronos-bolt",
        "parameter_count":sum(p.numel() for p in model.parameters()),"training_seconds":seconds,"best_epoch":best_epoch,
        "selection_metric":"validation mean pinball loss; test never selects checkpoints",
        "pretrained_validation":pretrained_validation,"labels":BANDS,"split_counts":{k:len(v) for k,v in parts.items()},
        "split_months":{k:sorted(v.target_month.unique().tolist()) for k,v in parts.items()},
        "baseline":{"name":"previous two-month average","mae":float(mean_absolute_error(test.target,test.baseline_amount)),
                    "macro_f1":float(f1_score(test.label,spending_bands(test.baseline_amount),labels=BANDS,average="macro",zero_division=0))},
        "data_source":"user CSV" if args.csv else "synthetic demo",
        "limitations":"Only two monthly observations per input. Demo future amounts have little predictable structure. Same cards recur across temporal splits; July test targets were previously inspected for DistilBERT. Real external validation is required. No demographic inference. Quantile coverage is empirical, not guaranteed.",
        "postprocessing":"Clip purchases to nonnegative, sort quantiles, interpolate p10/p50/p90; USD bands <200/<400/otherwise."})
    (out/"evaluation.json").write_text(json.dumps(evaluation,indent=2))
    (out/"training_history.json").write_text(json.dumps(history,indent=2))
    (out/"forecast_contract.json").write_text(json.dumps({"context_months":2,"horizon_months":1,"currency":"USD","bands":[200,400],"amount":"positive purchases","chronos_version":"2.3.2"},indent=2))
    again = BaseChronosPipeline.from_pretrained(str(out),device_map="cpu",torch_dtype=torch.float32)
    first = predict_amounts(reloaded, test.context.iloc[:2].tolist())
    second = predict_amounts(again, test.context.iloc[:2].tolist())
    assert np.allclose(first[["p10","median_amount","p90"]],second[["p10","median_amount","p90"]],atol=1e-3)
    (out/"reload_verification.json").write_text(json.dumps({"passed":True,"samples":2,"atol":1e-3}))
    (out/"README.md").write_text(f"---\nlibrary_name: chronos\nlicense: apache-2.0\npipeline_tag: time-series-forecasting\nbase_model: {BASE_FORECAST_MODEL}\n---\n# SpendLens spending forecast — Chronos-Bolt Tiny\n\nFine-tuned on {evaluation['data_source']}; not a production-validated financial model. No demographic inference.\n\nInput: two complete monthly positive purchase totals per card, in USD. Output: next-month p10/p50/p90 and derived LOW/MEDIUM/HIGH bands. Labels thresholds: 200 and 400 USD.\n\nBest epoch: {best_epoch}, selected on validation pinball loss. Test MAE: {evaluation['test']['test_mae']:.4f}; test band Macro-F1: {evaluation['test']['test_macro_f1']:.4f}; baseline MAE: {evaluation['baseline']['mae']:.4f}; baseline band Macro-F1: {evaluation['baseline']['macro_f1']:.4f}.\n\n{evaluation['limitations']}\n\nInstall `chronos-forecasting==2.3.2` and `transformers==4.57.6`. Load with `BaseChronosPipeline.from_pretrained(repo_id, device_map='cpu')`; call `predict(torch.tensor([[250., 320.]]), prediction_length=1)`. Output shape is batch × quantiles × horizon. See forecast_contract.json and evaluation.json.\n\nTraining script: https://github.com/zaynfuture/isom-group-project/blob/main/scripts/train_spendlens_chronos.py\n")
    print(json.dumps(evaluation,indent=2)); print("Saved and reload-verified", out,flush=True)


if __name__ == "__main__":
    main()
