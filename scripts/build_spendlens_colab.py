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
    cell("code", "!git clone https://github.com/zaynfuture/isom-group-project.git /content/isom-group-project\n%cd /content/isom-group-project\n%pip install -r requirements.txt\n"),
    cell("markdown", "## Inspect transactions\nThe reviewed Citi mapping currently covers five MCCs. Other codes stay unmapped.\n"),
    cell("code", "import sys\nsys.path.insert(0, 'src')\nfrom isom_project.spending import demo_transactions, monthly_examples\ndata = demo_transactions()\ndisplay(data)\ndisplay(monthly_examples(data))\n"),
    cell("markdown", "## Fine-tune both tasks from Hugging Face pretrained DistilBERT\nMerchant labels are derived from known MCCs; model inputs exclude MCC. Forecast inputs contain two prior months, never the future target month.\n"),
    cell("code", "!python scripts/train_spendlens.py --task merchant --epochs 3\n"),
    cell("code", "!python scripts/train_spendlens.py --task forecast --epochs 3\n"),
    cell("markdown", "## Review evaluation before upload\nCompare each test Macro-F1 against its baseline. Train scores are in-sample; synthetic results require external validation.\n"),
    cell("code", "import json\nfrom pathlib import Path\nfor task in ['merchant', 'forecast']:\n    print(task)\n    display(json.loads(Path(f'artifacts/spendlens_{task}/evaluation.json').read_text()))\n"),
    cell("markdown", "## Upload model files to zhengzhihust\nEnable explicitly after reviewing evaluation. Authentication is interactive; raw transactions are excluded.\n"),
    cell("code", "PUBLISH = False\nif PUBLISH:\n    from huggingface_hub import notebook_login\n    import subprocess\n    notebook_login()\n    for task in ['merchant', 'forecast']:\n        subprocess.run([sys.executable, 'scripts/upload_spendlens.py', '--task', task], check=True)\n"),
    cell("markdown", "## Configure Streamlit after successful uploads\nUse these root-level secrets:\n```toml\nSPENDLENS_MERCHANT_MODEL = \"zhengzhihust/spendlens-merchant-classifier\"\nSPENDLENS_FORECAST_MODEL = \"zhengzhihust/spendlens-spending-forecast\"\n```\nApplication code stays in GitHub. Models, tokenizers and evaluation evidence stay in Hugging Face. Verify both inference paths after deployment.\n"),
]
nb = dict(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},nbformat=4,nbformat_minor=5)
(Path(__file__).resolve().parents[1]/"SpendLens_Colab.ipynb").write_text(json.dumps(nb,indent=2))
