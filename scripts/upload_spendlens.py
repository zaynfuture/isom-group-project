"""Upload only model artifacts and evaluation evidence after explicit CLI invocation."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["merchant", "forecast"])
    args = parser.parse_args()
    suffix = "merchant-classifier" if args.task == "merchant" else "spending-forecast"
    repo = f"zhengzhihust/spendlens-{suffix}"
    folder = Path(__file__).resolve().parents[1] / "artifacts" / f"spendlens_{args.task}"
    for name in ["model.safetensors","config.json","tokenizer_config.json","evaluation.json","README.md"]:
        if not (folder/name).is_file():
            raise ValueError(f"Missing {name}; train and evaluate before uploading.")
    json.loads((folder/"evaluation.json").read_text())
    api = HfApi()
    if api.whoami()["name"].lower() != "zhengzhihust":
        raise ValueError("Authenticate as zhengzhihust before uploading.")
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo,folder_path=str(folder),allow_patterns=["*.json","*.safetensors","vocab.txt","README.md"],ignore_patterns=["checkpoints/**"])
    print(f"Uploaded model/tokenizer and evidence: https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
