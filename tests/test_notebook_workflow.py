"""Structural guardrails complement (not replace) executed notebook evidence."""
from pathlib import Path
import nbformat
import pytest

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('name',['01_merchant_experiment','02_forecast_experiment'])
def test_notebook_has_executable_lifecycle_without_hidden_training_cli(name):
    notebook=nbformat.read(ROOT/'notebooks'/f'{name}.ipynb',as_version=4)
    nbformat.validate(notebook)
    code='\n'.join(cell.source for cell in notebook.cells if cell.cell_type=='code')
    assert 'PUBLISH = False' in code
    assert 'upload_folder' in code
    assert 'reload_verification.json' in code
    assert 'evaluation.json' in code
    assert 'from_pretrained' in code
    assert 'model.training' not in code
    assert '!python' not in code
    for cell in notebook.cells:
        if cell.cell_type=='code': compile(cell.source,name,'exec')
