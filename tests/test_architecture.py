import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_model_has_no_app_or_streamlit_dependency():
    for path in (ROOT/"model").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = [alias.name for alias in node.names] if isinstance(node,ast.Import) else [node.module or ""] if isinstance(node,ast.ImportFrom) else []
            assert not any(name.split(".")[0] in {"streamlit","app"} for name in names),str(path)


def test_app_never_imports_training():
    for path in (ROOT/"app").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                assert not (node.module or "").startswith(("model.training","model.registry.upload")),str(path)


def test_registry_and_compatibility():
    import sys
    sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"src"))
    from model.inference.registry import deployed
    from isom_project.spending import demo_transactions as old
    from model.data.transactions import demo_transactions as new
    assert old is new
    assert deployed("merchant")["hub_reload_verified"]
    assert (ROOT/"model/artifacts"/deployed("forecast")["artifact_dir"]/"evaluation.json").exists()
