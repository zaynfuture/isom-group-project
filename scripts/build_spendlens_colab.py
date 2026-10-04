"""Build the navigation entry point for the complete experiment notebooks."""
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parents[1]
nb = nbformat.v4.new_notebook()
nb.cells = [nbformat.v4.new_markdown_cell('''# SpendLens experiment notebooks

The complete model-selection, fine-tuning, evaluation and publication code now lives in two dedicated notebooks:

1. [Merchant experiment — open in Colab](https://colab.research.google.com/github/zaynfuture/isom-group-project/blob/main/notebooks/01_merchant_experiment.ipynb)
2. [Forecast experiment — open in Colab](https://colab.research.google.com/github/zaynfuture/isom-group-project/blob/main/notebooks/02_forecast_experiment.ipynb)

For local Jupyter, open these files under `notebooks/`. See `docs/notebooks.md` for setup and evaluation contracts.

Both default to small smoke experiments, with Hugging Face publication disabled. Full experiments require an explicit configuration change. Upload and serving promotion are separate actions; no production manifest is automatically replaced.

This file is a navigation index, not an executed training report. Synthetic results are not production validation.
''')]
nb.cells[0]['id'] = 'spendlens-experiment-index'
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nbformat.write(nb, root / 'SpendLens_Colab.ipynb')
