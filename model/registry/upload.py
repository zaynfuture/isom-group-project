"""Upload only model artifacts and evaluation evidence after explicit CLI invocation."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["merchant", "forecast"])
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--repo-id")
    args = parser.parse_args()
    suffix = "merchant-classifier" if args.task == "merchant" else "spending-forecast"
    repo = args.repo_id or f"zhengzhihust/spendlens-{suffix}"
    if not repo.startswith("zhengzhihust/") or repo.count("/") != 1:
        raise ValueError("Uploads are restricted to the authorized zhengzhihust profile.")
    folder = args.folder or Path(__file__).resolve().parents[2] / "model" / "artifacts" / f"spendlens_{args.task}"
    config = json.loads((folder/"config.json").read_text())
    required = ["model.safetensors","config.json","evaluation.json","README.md","reload_verification.json"]
    required += ["forecast_contract.json"] if "chronos_config" in config else ["tokenizer_config.json"]
    for name in required:
        if not (folder/name).is_file():
            raise ValueError(f"Missing {name}; train and evaluate before uploading.")
    json.loads((folder/"evaluation.json").read_text())
    if json.loads((folder/"reload_verification.json").read_text()).get("passed") is not True:
        raise ValueError("Model must pass save/reload verification before publication.")
    api = HfApi()
    if api.whoami()["name"].lower() != "zhengzhihust":
        raise ValueError("Authenticate as zhengzhihust before uploading.")
    api.create_repo(repo, repo_type="model", exist_ok=True)
    receipt = api.upload_folder(repo_id=repo,folder_path=str(folder),allow_patterns=["config.json","model.safetensors","tokenizer.json","tokenizer_config.json","special_tokens_map.json","evaluation.json","training_history.json","forecast_contract.json","reload_verification.json","vocab.txt","README.md"],ignore_patterns=["checkpoints/**"])
    (folder/"upload_receipt.json").write_text(json.dumps({"repo_id":repo,"revision":receipt.oid},indent=2))
    print(f"Uploaded model/tokenizer and evidence: https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
