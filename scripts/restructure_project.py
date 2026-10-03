"""One-time mechanical app/model split; run only against this repository."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
MOVES = {
    "spendlens_app.py":"app/main.py",
    "src/isom_project/spending.py":"model/data/transactions.py",
    "src/isom_project/mcc_reference.py":"model/data/mcc_reference.py",
    "src/isom_project/spend_models.py":"model/inference/registry.py",
    "src/isom_project/forecasting.py":"model/inference/forecasting.py",
    "src/isom_project/spending_fairness.py":"app/services/disparity.py",
    "src/isom_project/fairness_page.py":"app/views/fairness.py",
    "scripts/train_spendlens.py":"model/training/merchant.py",
    "scripts/train_spendlens_chronos.py":"model/training/chronos.py",
    "scripts/upload_spendlens.py":"model/registry/upload.py",
    "scripts/verify_spendlens_hub.py":"model/registry/verify.py",
}
REPLACEMENTS = {
    "from isom_project.spending import":"from model.data.transactions import",
    "from isom_project.spend_models import":"from model.inference.registry import",
    "from isom_project.forecasting import":"from model.inference.forecasting import",
    "from isom_project.mcc_reference import":"from model.data.mcc_reference import",
    "from isom_project.fairness_page import":"from app.views.fairness import",
    "from .spending_fairness import":"from app.services.disparity import",
}


def main():
    for folder in ["app","app/views","app/services","model","model/data","model/inference","model/training","model/registry"]:
        path = ROOT/folder
        path.mkdir(parents=True,exist_ok=True)
        init = path/"__init__.py"
        if not init.exists():
            init.write_text('"""SpendLens component package."""\n')
    for old,new in MOVES.items():
        source,dest = ROOT/old,ROOT/new
        if dest.exists():
            raise ValueError(f"Refusing to overwrite {dest}; migration already applied.")
        content = source.read_text()
        for before,after in REPLACEMENTS.items():
            content = content.replace(before,after)
        if new.startswith(("model/training/","model/registry/")):
            content = content.replace("parents[1]", "parents[2]")
            content = content.replace('ROOT / "src"', 'ROOT').replace('ROOT/"src"','ROOT')
        if new == "app/main.py":
            content = content.replace('Path(__file__).resolve().parent / "src"','Path(__file__).resolve().parents[1]')
            content = content.replace('ROOT / "artifacts"','ROOT / "model" / "artifacts"')
        if new.startswith("model/"):
            content = content.replace('ROOT / "artifacts"','ROOT / "model" / "artifacts"')
            content = content.replace('ROOT/"artifacts', 'ROOT/"model/artifacts')
            content = content.replace('parents[2] / "artifacts"','parents[2] / "model" / "artifacts"')
        dest.write_text(content)
        module = new.removesuffix(".py").replace("/", ".")
        if old.startswith("src/"):
            source.write_text(f'"""Compatibility import; implementation moved to {module}."""\nimport sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parents[2]))\nfrom {module} import *  # noqa: F403\n')
        elif old.startswith("scripts/"):
            source.write_text(f'"""Compatibility command; implementation moved to {module}."""\nimport sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parents[1]))\nfrom {module} import main\nif __name__ == "__main__":\n    main()\n')
        else:
            source.write_text('"""Compatibility Streamlit entry point."""\nimport runpy\nfrom pathlib import Path\nrunpy.run_path(str(Path(__file__).resolve().parent/"app/main.py"),run_name="__main__")\n')
    artifacts = ROOT/"model/artifacts"
    artifacts.mkdir(exist_ok=True)
    for name in ["spendlens_deployment.json","spendlens_merchant","spendlens_forecast","spendlens_merchant_minilm","spendlens_forecast_chronos"]:
        source = ROOT/"artifacts"/name
        if source.exists():
            shutil.move(str(source),str(artifacts/name))


if __name__ == "__main__":
    main()
