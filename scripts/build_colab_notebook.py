"""Build the upload-ready Google Colab notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ISOM_Group_Project_Colab.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "accelerator": "GPU",
    "colab": {
        "name": OUT.name,
        "provenance": [],
        "toc_visible": True,
        "gpuType": "T4",
    },
}


def md(text: str) -> None:
    nb.cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    nb.cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# FairnessLens: Hugging Face Fine-Tuning and Evaluation

**ISOM5240 Group Project**  
**Business objective:** determine whether demographics imputed from proxy text preserve the fairness conclusions obtained from known benchmark groups.

This notebook trains the model used by the Streamlit application. It uses only synthetic Group A/Group B data and must not be interpreted as a real demographic classifier.

## Before running

1. In Colab, choose **Runtime → Change runtime type → T4 GPU**.
2. Upload `isom-group-project.zip` when prompted below.
3. Run all cells in order.

No Hugging Face token is required for the public default checkpoint.
""")

code("""
from pathlib import Path
import os
import platform
import shutil
import subprocess
import sys
import zipfile

IN_COLAB = "google.colab" in sys.modules
print("Environment:", "Google Colab" if IN_COLAB else "Local Jupyter")
print("Python:", platform.python_version())

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except ImportError:
    print("PyTorch will be installed in the next step.")
""")

md("""
## 1. Locate or upload the project

If `/content/isom-group-project` does not exist, this cell opens Colab's file uploader. Select `isom-group-project.zip`. Local Jupyter users should launch the notebook from the project root.
""")

code("""
if IN_COLAB:
    project_root = Path("/content/isom-group-project")
    if not project_root.exists():
        from google.colab import files
        uploaded = files.upload()
        zip_names = [name for name in uploaded if name.lower().endswith(".zip")]
        if not zip_names:
            raise FileNotFoundError("Upload isom-group-project.zip and rerun this cell.")
        with zipfile.ZipFile(zip_names[0]) as archive:
            archive.extractall("/content")
else:
    project_root = Path.cwd()
    if project_root.name != "isom-group-project":
        candidate = project_root / "isom-group-project"
        if candidate.exists():
            project_root = candidate

if not (project_root / "app.py").exists():
    raise FileNotFoundError(f"Project not found at {project_root.resolve()}")

os.chdir(project_root)
print("Project root:", project_root.resolve())
""")

md("""
## 2. Install the reproducible environment

The restart warning sometimes shown by Colab can be ignored unless an import fails. The project versions are constrained in `requirements-colab.txt`.
""")

code("""
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"
])
print("Dependencies installed.")
""")

code("""
# Make the local package importable in this runtime.
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
import pandas as pd
import sklearn
import torch
import transformers

print({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit-learn": sklearn.__version__,
})
""")

md("""
## 3. Generate the privacy-preserving demonstration data

The generator creates overlapping neutral proxy tokens, known synthetic groups, qualification status, and decisions. A direct synthetic group penalty creates a measurable downstream disparity. The random seed makes every run reproducible.
""")

code("""
subprocess.check_call([sys.executable, "scripts/generate_data.py"])
data = pd.read_csv("data/generated/synthetic_audit_data.csv")
display(data.head())
display(data.groupby("split")[["group_b", "qualified", "decision"]].agg(["count", "mean"]))
""")

md("""
## 4. Fine-tune the Hugging Face model

The default `google/bert_uncased_L-2_H-128_A-2` checkpoint is a compact, pretrained two-layer BERT. It is substantially more resource-efficient than full BERT while retaining the standard Transformer architecture. Model selection is based on validation macro-F1; the test split remains held out until training is complete.
""")

code("""
MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
EPOCHS = 4
BATCH_SIZE = 32

command = [
    sys.executable,
    "scripts/train.py",
    "--model-name", MODEL_NAME,
    "--epochs", str(EPOCHS),
    "--batch-size", str(BATCH_SIZE),
]
print("Running:", " ".join(command))
subprocess.check_call(command)
""")

md("""
## 5. Evaluate classification and audit validity

The evaluation intentionally has two levels:

- **Classification:** accuracy, balanced accuracy, macro precision/recall/F1, ROC AUC, Brier score, and confusion matrix.
- **Downstream audit:** observed versus soft-imputed demographic parity, impact ratio, equal opportunity, false-positive-rate gap, and absolute error.

A model can perform well at the first level and still fail at the second.
""")

code("""
subprocess.check_call([sys.executable, "scripts/evaluate.py"])

import json
metrics = json.loads(Path("artifacts/metrics.json").read_text())
classification = pd.Series(metrics["classification"]).drop("confusion_matrix")
fairness = pd.DataFrame(metrics["fairness"]).T

display(classification.to_frame("value").style.format("{:.4f}"))
display(pd.DataFrame(metrics["classification"]["confusion_matrix"],
                     index=["True A", "True B"],
                     columns=["Predicted A", "Predicted B"]))
display(fairness.style.format("{:.4f}"))
""")

code("""
import matplotlib.pyplot as plt

plot_data = fairness.loc[
    ["Observed group", "Imputed (soft)"],
    ["demographic_parity_gap", "equal_opportunity_gap", "false_positive_rate_gap"],
].T
ax = plot_data.plot.bar(figsize=(10, 5), color=["#173F5F", "#4AA3DF"])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Group B minus Group A")
ax.set_title("Observed and imputed fairness estimates")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()
""")

md("""
## 6. Verify the exact model interface used by Streamlit

This calls the same loading and prediction functions used by `app.py`, proving that the saved artifact can be consumed by the application.
""")

code("""
from isom_project.config import MODEL_DIR
from isom_project.modeling import load_pipeline, predict_probabilities

classifier = load_pipeline(MODEL_DIR)
examples = [
    "profile token alden; region north; channel web; product credit; income band 3; tenure 5 years",
    "profile token pavan; region south; channel branch; product savings; income band 4; tenure 8 years",
]
probabilities = predict_probabilities(classifier, examples)
pd.DataFrame({"text": examples, "P(Group B)": probabilities})
""")

md("""
## 7. Optional: upload the saved model to Hugging Face Hub

After fine-tuning and evaluation, upload the saved model and tokenizer to zhengzhihust/fairnesslens-demographic-imputer. Set `PUSH_TO_HUB=True` when ready and authenticate interactively with that account. No credentials are stored in this notebook.
""")

code("""
PUSH_TO_HUB = False
HF_REPO_ID = "zhengzhihust/fairnesslens-demographic-imputer"

if PUSH_TO_HUB:
    if not HF_REPO_ID or "/" not in HF_REPO_ID:
        raise ValueError("Set HF_REPO_ID to YOUR_ACCOUNT/MODEL_NAME")
    from huggingface_hub import notebook_login
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    notebook_login()
    saved_model = AutoModelForSequenceClassification.from_pretrained("artifacts/model")
    saved_tokenizer = AutoTokenizer.from_pretrained("artifacts/model")
    saved_model.push_to_hub(HF_REPO_ID)
    saved_tokenizer.push_to_hub(HF_REPO_ID)
    print(f"Uploaded to https://huggingface.co/{HF_REPO_ID}")
else:
    print("Hub upload skipped. Set PUSH_TO_HUB=True when ready.")
""")

md("""
## 8. Package the trained model and results

Download the ZIP, extract it into the local project's `artifacts/` directory, and run `streamlit run app.py`. The Streamlit app then uses this exact fine-tuned model for single-record and batch inference.
""")

code("""
bundle = Path("/content/fairnesslens-trained-artifacts.zip") if IN_COLAB else project_root / "fairnesslens-trained-artifacts.zip"
if bundle.exists():
    bundle.unlink()
with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in (project_root / "artifacts").rglob("*"):
        if path.is_file() and "trainer" not in path.parts:
            archive.write(path, path.relative_to(project_root))
print("Created:", bundle)

if IN_COLAB:
    from google.colab import files
    files.download(str(bundle))
""")

md("""
## Interpretation and responsible use

Do not select the model solely because of macro-F1. Compare its inferred-group fairness metrics with the known-group benchmark and report the error. The synthetic experiment demonstrates a validation method, not real-world demographic validity. Inferred attributes should never determine individual eligibility, pricing, service, or enforcement decisions.
""")

nbf.write(nb, OUT)
print(f"Wrote {OUT}")
