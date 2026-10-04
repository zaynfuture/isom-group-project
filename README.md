# SpendLens

SpendLens is an English-language application for enterprise card-spending analytics, merchant classification, spending forecasts and group-disparity analysis.

[Live application](https://spendlens-card-analytics.streamlit.app/) · [Notebook guide](docs/notebooks.md) · [Deployment](docs/deployment.md)

## Run the application

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Community Cloud continues to run `app.py` using the root runtime requirements. A standalone React/API migration was not adopted because preserving this deployment is a requirement. Notebook dependencies are separate from the deployed application.

## User journey

1. Load the synthetic example or upload transaction data and review validation results.
2. Explore purchases, refunds, net spending, categories and trends.
3. Run merchant classification or next-month purchase forecasting and inspect model evidence.
4. Compare spending across groups when authorized demographic labels are available. These comparisons describe disparities, not proof of discrimination; the application does not infer personal ethnicity or gender from spending.

Reporting defaults to USD. The application offers mainstream currencies and an expanded provider catalog, with reference-rate conversion and retained audit fields. Forecast contracts remain USD-based; currency conversion is not a settlement-rate guarantee.

## Data and model scope

The application demonstration contains 13,213 synthetic transactions for 120 cards: 12,747 purchases, 294 full refunds and 172 partial refunds. Fourteen categories and six lifestyle scenarios exercise application behavior. These are simulation assumptions, not measured population behavior.

The separate legacy training fixture contains 7,680 transactions and five merchant categories. The currently deployed classifier recognizes only these five trained labels. Broader MCC analytics coverage does not expand the classifier's label schema. Raw MCCs and mapping information remain available; unsupported codes must not be silently relabeled.

## Complete notebook experiments

- [Merchant model selection, fine-tuning and evaluation](notebooks/01_merchant_experiment.ipynb): compare pretrained BERT Tiny and MiniLM encoders using card-disjoint training/validation/test partitions. Select on validation macro-F1, then evaluate the chosen model on the test partition.
- [Spending forecast selection, fine-tuning and evaluation](notebooks/02_forecast_experiment.ipynb): compare Chronos-Bolt Tiny and Mini on validation pinball loss, fine-tune the selected model, and evaluate chronological held-out forecasts against a simple spending baseline.

Both notebooks contain actual training code, metrics, plots, checkpoint saving, local reload checks, model-card generation and an explicitly gated Hugging Face upload/reload section. They are not wrappers around a training CLI.

```sh
pip install -r requirements-notebook.txt
jupyter lab notebooks/
```

Default execution is a small smoke run (`SPENDLENS_SMOKE=1`). Set `SPENDLENS_SMOKE=0` before executing a full experiment. `SPENDLENS_DATASET` selects `legacy` or `lifestyle`. Each execution writes an independent directory below `outputs/notebook-runs/`; provenance includes data hash, candidate model revisions and evaluation results. Smoke results are execution checks, not model-quality claims.

Publication defaults to `PUBLISH=False`. A reviewed full run can publish a new experiment repository under `zhengzhihust`; notebooks protect the existing production repository names and never automatically update the deployment manifest. See the notebook guide for review and promotion boundaries.

## Architecture

```text
app.py                  Streamlit Cloud entry point
app/                    User interface and application services
model/data/             Transaction contracts and synthetic fixtures
model/inference/        Version-pinned serving and prediction contracts
model/experiments.py    Notebook split contracts
model/training/         Existing training CLI utilities
notebooks/              Visible experiment lifecycle
model/artifacts/        Committed historical serving evidence
tests/                  Public-contract and application tests
docs/                   Workflow, deployment and decision records
```

The deployment manifest is `model/artifacts/spendlens_deployment.json`. Existing Hub models are [merchant MiniLM](https://huggingface.co/zhengzhihust/spendlens-merchant-minilm) and [Chronos-Bolt Tiny](https://huggingface.co/zhengzhihust/spendlens-spending-chronos-bolt-tiny). New notebook runs do not replace their pinned revisions or historical evaluation evidence.

## Verification and engineering workflow

```sh
pip install -r requirements-dev.txt
python -m pytest -q
python scripts/build_experiment_notebooks.py
python scripts/verify_experiment_notebooks.py 01_merchant_experiment
python scripts/verify_experiment_notebooks.py 02_forecast_experiment
SPENDLENS_RUN_MODEL_TESTS=1 python -m pytest -q tests/test_serving_integration.py
```

The execution checks download public pretrained models and perform small real CPU experiments. Reports and executed notebooks are written to `outputs/notebook-checks/`. Structural notebook tests alone do not prove that training executes successfully.

The real-weight Streamlit test is opt-in; use `HF_HUB_OFFLINE=1` when the pinned serving weights are already cached. Committed [smoke evidence](docs/verification/notebook-smoke.json) records the verified local runs, not full-run model quality or Colab-hosted execution.

See [AGENTS.md](AGENTS.md), [domain vocabulary](GLOSSARY.md) and [architecture decision](docs/adr/0001-streamlit-and-notebook-lifecycle.md). GitHub Issues is the agreed tracker. If write access is unavailable, the [unpublished issue draft](docs/issue-drafts/notebook-lifecycle.md) remains explicitly pending; it is not an existing GitHub issue.

The workflow follows selected recommendations from [mattpocock/skills](https://github.com/mattpocock/skills), reviewed at commit `d81f3a183412e71a5b1e84ca21bc1a35eea03a60`: repository-specific agent instructions, explicit domain terminology, decision records and tests at public interfaces. Existing specifications and historical FairnessLens materials are retained as history, not evidence of current experiment performance.
