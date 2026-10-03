"""Build the explicit user-run Colab training and deployment workflow."""
import json
from pathlib import Path
def cell(kind, text):
    result = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True))
    if kind == "code":
        result.update(execution_count=None,outputs=[])
    return result
cells = [
    cell("markdown", "# SpendLens: pretrained models → fine-tuning → evaluation → Hugging Face\nUse a Colab GPU runtime. Training begins only when you execute the training cells. Demo data is synthetic. The two checkpoints are separate.\n"),
    cell("code", "!git clone https://github.com/zaynfuture/isom-group-project.git /content/isom-group-project\n%cd /content/isom-group-project\n%pip install -r requirements-colab.txt\n"),
    cell("markdown", "## Inspect transactions\nThe reviewed Citi mapping currently covers five MCCs. Other codes stay unmapped.\n"),
    cell("code", "import sys\nsys.path.insert(0, 'src')\nfrom model.data.transactions import demo_transactions, monthly_examples\ndata = demo_transactions()\ndisplay(data)\ndisplay(monthly_examples(data))\n"),
    cell("markdown", "## Fine-tune task-appropriate pretrained models\nMiniLM classifies merchant descriptions; MCC is excluded from input. Chronos-Bolt forecasts numeric purchase totals from two previous months. Use fresh output directories for reruns. Demo data is not representative business validation.\n"),
    cell("code", "!python -m model.training.merchant --task merchant --model-name microsoft/MiniLM-L12-H384-uncased --epochs 3 --output-dir model/artifacts/spendlens_merchant_minilm\n"),
    cell("code", "!python -m model.training.chronos --epochs 5 --output-dir model/artifacts/spendlens_forecast_chronos\n"),
    cell("markdown", "## Review evaluation before upload\nCompare each test Macro-F1 against its baseline. Train scores are in-sample; synthetic results require external validation.\n"),
    cell("code", "import json\nfrom pathlib import Path\nfor folder in ['spendlens_merchant_minilm', 'spendlens_forecast_chronos']:\n    print(folder)\n    display(json.loads(Path(f'model/artifacts/{folder}/evaluation.json').read_text()))\n"),
    cell("markdown", "## Upload model files to zhengzhihust\nEnable explicitly after reviewing evaluation. Authentication is interactive; raw transactions are excluded.\n"),
    cell("code", "PUBLISH = False\nif PUBLISH:\n    from huggingface_hub import notebook_login\n    import subprocess\n    notebook_login()\n    jobs = [('merchant', 'spendlens_merchant_minilm', 'spendlens-merchant-minilm'), ('forecast', 'spendlens_forecast_chronos', 'spendlens-spending-chronos-bolt-tiny')]\n    for task, folder, repo in jobs:\n        subprocess.run([sys.executable, '-m', 'model.registry.upload', '--task', task, '--folder', f'model/artifacts/{folder}', '--repo-id', f'zhengzhihust/{repo}'], check=True)\n    subprocess.run([sys.executable, '-m', 'model.registry.verify'], check=True)\n"),
    cell("markdown", "## Deploy only verified revisions\nThe verification script reloads uploaded checkpoints at their commit SHA, checks inference and writes model/artifacts/spendlens_deployment.json. Commit that manifest plus evaluation/training-history evidence to GitHub; do not commit weights or credentials. Streamlit loads pinned Hub revisions. Review any existing SPENDLENS model environment overrides before deployment. Forecast scores are numeric quantiles, not class probabilities.\n"),
]
nb = dict(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},nbformat=4,nbformat_minor=5)
(Path(__file__).resolve().parents[1]/"SpendLens_Colab.ipynb").write_text(json.dumps(nb,indent=2))
