"""Reload pinned published weights, compare local predictions, write serving manifest."""
import json
import os
os.environ.setdefault("USE_TF", "0")
from pathlib import Path
import sys
import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import pipeline
from chronos import BaseChronosPipeline
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from model.inference.forecasting import predict_amounts


def main():
    torch.set_num_threads(2)
    manifest = {}
    for task, directory, architecture in [("merchant","spendlens_merchant_minilm","minilm"),("forecast","spendlens_forecast_chronos","chronos-bolt")]:
        folder = ROOT/"model/artifacts"/directory
        receipt = json.loads((folder/"upload_receipt.json").read_text())
        repo, revision = receipt["repo_id"], receipt["revision"]
        info = HfApi().model_info(repo,revision=revision)
        assert info.sha == revision
        assert not info.private, "Expected a publicly accessible course model repository."
        if task == "merchant":
            remote = pipeline("text-classification",model=repo,tokenizer=repo,revision=revision,device=-1,top_k=None,framework="pt")
            local = pipeline("text-classification",model=str(folder),tokenizer=str(folder),device=-1,top_k=None,framework="pt")
            inputs = ["Fresh Market 12","Harbor Hotel 8"]
            a,b = remote(inputs),local(inputs)
            for remote_row,local_row in zip(a,b):
                remote_scores = {v["label"]:v["score"] for v in remote_row}
                local_scores = {v["label"]:v["score"] for v in local_row}
                assert remote_scores.keys() == local_scores.keys()
                assert np.allclose(list(remote_scores.values()),[local_scores[k] for k in remote_scores],atol=1e-6)
            demo = [{"label":max(row,key=lambda x:x["score"])["label"]} for row in a]
        else:
            remote = BaseChronosPipeline.from_pretrained(repo,revision=revision,device_map="cpu",torch_dtype=torch.float32)
            local = BaseChronosPipeline.from_pretrained(str(folder),device_map="cpu",torch_dtype=torch.float32)
            inputs = [[250.,320.],[450.,410.]]
            a,b = predict_amounts(remote,inputs),predict_amounts(local,inputs)
            assert np.allclose(a[["p10","median_amount","p90"]],b[["p10","median_amount","p90"]],atol=1e-6)
            demo = a.to_dict("records")
        print(json.dumps({"task":task,"repo_id":repo,"revision":revision,"verified":True,"demo":demo}),flush=True)
        manifest[task] = {**receipt,"architecture":architecture,"artifact_dir":directory,"hub_reload_verified":True}
        del remote,local
    (ROOT/"model/artifacts/spendlens_deployment.json").write_text(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    main()
