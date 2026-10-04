# isom-group-project

## Current project: SpendLens

### Application and model architecture

`app/` owns the English Streamlit interface and currency/disparity services. `model/` owns transaction contracts, inference, training, Hugging Face publication and evaluation artifacts. Training dependencies are separate from app deployment dependencies; the root `app.py` remains the Streamlit Cloud launcher. See [architecture and currency specification](specs/005-architecture-currency.md).

Reporting defaults to **USD**. Choose from 50 mainstream currencies or enable the expanded provider catalog. `CurrencyConverter` uses historical Frankfurter reference rates to normalize mixed-currency transactions before aggregation; original amounts and rate audit fields are retained. Missing/stale rates stop conversion instead of silently mixing currencies. Forecast inputs and spending-band thresholds stay in USD, with optional converted display amounts.

Live application: https://spendlens-card-analytics.streamlit.app/

The main Streamlit entry point now serves card spending analytics. Run `streamlit run app.py`. Previous FairnessLens code remains in `legacy_fairness_app.py`; its results must not be presented as SpendLens evidence.

- Browse 13,213 synthetic records for 120 cards across eight months: 12,747 purchases, 294 full refunds and 172 partial refunds. Fourteen categories and six balanced lifestyle scenarios support varied activity, recurring bills and weekday/weekend behavior. These are simulation assumptions, not real-world distribution estimates. The original 7,680-row training fixture remains separate; see `specs/007-behavioral-synthetic-data.md`.
- View currency-normalized purchases, refunds, net spending and category trends.
- Pipeline 1: pretrained Microsoft MiniLM fine-tuned for merchant category classification.
- Pipeline 2: pretrained Amazon Chronos-Bolt Tiny fine-tuned on numeric monthly purchases, producing next-month p10/p50/p90 amounts and derived spending bands.
- `SpendLens_Colab.ipynb` supplies reproducible training, evaluation, save/reload checks and explicit upload steps. Model Evidence shows committed run artifacts; never treat synthetic results as production validation.
- MCC reference: [user-supplied Citi manual](https://www.citibank.com/tts/solutions/commercial-cards/assets/docs/govt/Merchant-Category-Codes.pdf). Analytics recognizes 14 reviewed codes; the merchant model remains limited to its five trained labels. Raw MCC, normalized description and mapping version are retained. Other codes are explicitly unmapped.
- FairnessLens compares gender, age-band and ethnicity spending disparities using authorized labels, card-level permutation tests, Holm correction and bootstrap intervals. Statistical differences do not establish discrimination; identities are not inferred from spending.

Model repositories: [merchant MiniLM](https://huggingface.co/zhengzhihust/spendlens-merchant-minilm) and [spending Chronos-Bolt](https://huggingface.co/zhengzhihust/spendlens-spending-chronos-bolt-tiny). The application uses `model/artifacts/spendlens_deployment.json` to pin verified revisions and matching evaluation directories. Do not replace a verified revision with a moving branch or commit credentials. Existing environment overrides take precedence and must match the task architecture.

Reproduce in Colab or a compatible Python environment:

```sh
pip install -r requirements-colab.txt
python -m model.training.merchant --task merchant --model-name microsoft/MiniLM-L12-H384-uncased --epochs 3 --output-dir model/artifacts/spendlens_merchant_minilm
python -m model.training.chronos --epochs 5 --output-dir model/artifacts/spendlens_forecast_chronos
# Authenticate locally using the Hugging Face CLI or notebook_login (never put tokens in code).
python -m model.registry.upload --task merchant --folder model/artifacts/spendlens_merchant_minilm --repo-id zhengzhihust/spendlens-merchant-minilm
python -m model.registry.upload --task forecast --folder model/artifacts/spendlens_forecast_chronos --repo-id zhengzhihust/spendlens-spending-chronos-bolt-tiny
python -m model.registry.verify
```

Use fresh output directories for reruns: training scripts protect existing model weights. Chronos is trained directly with PyTorch using the official library, not a text tokenizer. Source code and evaluation JSON go to GitHub; weights go only to Hugging Face.

MiniLM synthetic test Macro-F1: **1.000**, equal to historical DistilBERT; 33.4M parameters. This is a footprint improvement, not demonstrated accuracy improvement. Chronos synthetic test: MAE **89.64 USD** vs baseline **92.15 USD**; band Macro-F1 **0.376** vs baseline **0.386** and historical DistilBERT **0.282**. Nominal 80% interval coverage: **78.3%**. This is mixed evidence, not a universal model improvement. Only two monthly observations are available per input and the synthetic generator has little predictive structure. Merchant templates overlap across card-held-out splits, so perfect template accuracy does not demonstrate unseen-merchant generalization. Test targets have been inspected in prior experiments.

Current scope: enterprise spending analytics, FairnessLens screening and task-specific fine-tuning/publication workflows. Report/PPT/video remain separate deliverables. Real financial-data use additionally requires controlled hosting, authentication, representative labeled data, longer account histories and approved data access; the public app uses synthetic demonstration data.

Specifications: `specs/002-spendlens.md`, `specs/003-fairnesslens-spending.md` and `specs/004-task-specific-pretraining.md`. Historical FairnessLens documentation below is retained for traceability, not as the current training instructions.

## FairnessLens: Demographic Imputation for Fairness Evaluation

### Course-aligned workflow

The fictional Meridian Financial Services case gives analysts measurable audit targets. Home presents retrospective acceptance targets; System Pipelines documents Transformer imputation and downstream statistical fairness auditing. These are two application workflows using one ML model, not two Hugging Face task types; the instructor's interpretation of “two pipelines” remains unconfirmed.

Start from Hugging Face `google/bert_uncased_L-2_H-128_A-2` pretrained weights and fine-tune later using the Colab notebook or `scripts/train.py`. The live app continues serving the previously fine-tuned model; no new training is triggered by app startup. The unfine-tuned base model is not a ready-to-use demographic classifier.

Model Evidence shows saved training and validation curves, checkpoint selection, evaluation on all three splits and the test confusion matrix. Training scores are in-sample. Test labels do not select checkpoints, but test results have been inspected across project iterations, so they are not a single-use external validation.

FairnessLens is a Streamlit business application that tests whether demographic attributes imputed from proxy text preserve downstream fairness conclusions. The project fine-tunes a Hugging Face transformer, evaluates both classification quality and fairness-estimation error, and deploys the resulting model in an interactive audit interface.

> **Responsible-use boundary:** the included data are synthetic and use neutral Group A/Group B labels. The application is a classroom validation prototype—not an identity classifier, production compliance tool, or basis for individual decisions.

## Why this meets the ISOM5240 group-project rubric

| Rubric area | Project evidence |
|---|---|
| Business App (25%) | Measurable objective; usable Streamlit workflow; CSV upload, validation, inference, fairness comparison, and download |
| Model (15%) | Hugging Face 2-layer BERT fine-tuning, probabilistic inference, resource-aware sequence length, and reproducible training |
| Expected Results (10%) | Held-out classification metrics plus observed-versus-imputed fairness error |
| Coding (20%) | Modular Python package, deterministic data generation, error handling, caching, and automated tests |
| Documentation & Presentation (10%) | This README, model limitations in the app, reproducible commands, and evaluation artifacts |

The syllabus states that a separate file contains detailed project instructions. That file was not supplied, so this implementation maps to the rubric available in `isom.pdf` and should be reconciled with the separate instructions when obtained.

## Architecture

```text
Synthetic proxy text + known audit labels
                |
                v
Hugging Face tokenizer -> fine-tuned compact BERT -> P(Group B)
                |                              |
                |                              v
                |                  classification evaluation
                v
soft probability weighting -> fairness metrics -> observed/imputed error
                                                   |
                                                   v
                                            Streamlit app
```

## Local setup

Python 3.10-3.12 is recommended.

```bash
cd /Users/zaynfuture/Developments/isom-group-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Generate data, fine-tune, evaluate, and launch:

```bash
python scripts/generate_data.py
python scripts/train.py --model-name google/bert_uncased_L-2_H-128_A-2 --epochs 4
python scripts/evaluate.py
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Google Colab training

The easiest route is the included `ISOM_Group_Project_Colab.ipynb`:

1. Open the notebook in Google Colab.
2. Select a T4 GPU runtime.
3. Upload `isom-group-project.zip` when prompted.
4. Run all cells.

The notebook installs dependencies, generates data, fine-tunes the model, evaluates both classification and fairness preservation, verifies the Streamlit inference interface, and downloads a trained-artifact ZIP.

The equivalent manual Colab commands are:

```python
%cd /content/isom-group-project
!pip install -q -r requirements-colab.txt
!python scripts/generate_data.py
!python scripts/train.py --model-name google/bert_uncased_L-2_H-128_A-2 --epochs 4 --batch-size 32
!python scripts/evaluate.py
```

Download the `artifacts/model/` directory and copy it to the same location in the local project before launching Streamlit. The default official 2-layer BERT checkpoint is small enough for a fast CPU or Colab run while still starting from meaningful pretrained weights. For a pipeline-only smoke test, use `--model-name hf-internal-testing/tiny-random-distilbert --epochs 1 --sample-limit 64`; that random checkpoint validates execution but is not a meaningful final model.

## Evaluation design

### Demographic-imputation quality

- Accuracy and balanced accuracy
- Macro precision, recall, and F1
- ROC AUC
- Brier score (probability calibration)
- Confusion matrix

### Downstream audit validity

- Group selection rates
- Demographic-parity gap
- Disparate-impact ratio
- Equal-opportunity gap
- False-positive-rate gap
- Absolute error versus the known synthetic group benchmark

The two levels are deliberately separate: a classifier can score well while still distorting the fairness audit.

## Course-aligned save, reload, and Hugging Face Hub flow

The training script follows the supplied course notebooks while strengthening their evaluation and reliability checks:

```bash
# Fine-tune, select the best checkpoint by validation macro-F1, save model + tokenizer,
# verify that reloaded logits match, and create artifacts/fairnesslens_model.zip.
python scripts/train.py

# Optional: authenticate first, then upload the saved model and tokenizer.
hf auth login
python scripts/train.py --push-to-hub --hub-model-id zhengzhihust/fairnesslens-demographic-imputer

# Run Streamlit directly from that Hub model rather than the bundled local model.
HF_MODEL_ID=zhengzhihust/fairnesslens-demographic-imputer streamlit run app.py
```

Unlike the classroom accuracy-only example, checkpoint selection uses macro-F1 and the saved metadata includes ROC AUC and Brier score. Dynamic padding avoids padding every example to the maximum length, early stopping limits overfitting, and the reload test catches incomplete model/tokenizer exports before deployment.

### Verified full-run results

The included final model was fine-tuned for four epochs on 4,200 synthetic training records and evaluated on 900 held-out records:

| Metric | Result |
|---|---:|
| Accuracy | 0.892 |
| Macro-F1 | 0.891 |
| ROC AUC | 0.977 |
| Brier score | 0.073 |
| Observed demographic-parity gap | -0.150 |
| Soft-imputed demographic-parity gap | -0.068 |
| Absolute DP-gap error | 0.082 |

The result is intentionally interpreted at both levels: the classifier discriminates the synthetic groups reasonably well, yet probability weighting substantially attenuates the downstream disparity. This is evidence for the project's central claim—not a production-valid demographic model.

## Streamlit pages

1. **Home:** product purpose, workflow, and required intended-use acknowledgement.
2. **Single Prediction:** inspect a probability from the fine-tuned model.
3. **Batch Audit:** validate or upload CSV data, compare fairness metrics, and export evidence.
4. **Model Evidence:** view held-out classification and downstream audit-validity results.
5. **Responsible Use:** intended use, prohibited use, limitations, and governance controls.

Uploaded CSV files require `text`, `qualified`, and `decision`. Include `group_b` when ground truth is available so the app can calculate fairness-estimation error.

## Spec-driven development

The product contract, implementation decisions, and delivery checklist live in
[`specs/001-fairness-audit-web-app`](specs/001-fairness-audit-web-app). The Streamlit
entry point is intentionally presentation-focused; reusable validation, scoring,
and report construction live in `src/isom_project/audit.py` and are covered by tests.

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub. Keep `app.py`, `requirements.txt`, `.python-version`,
   `.streamlit/config.toml`, and `artifacts/model/` in the repository.
2. In Streamlit Community Cloud, select **Create app**, choose the repository and
   branch, and set the main file to `app.py`.
3. Use Python 3.11. No secret is required when using the bundled model.
4. Optional: add `HF_MODEL_ID = "account/model-name"` in the app's Secrets settings
   to load a public Hugging Face model instead. Never commit access tokens.
5. Deploy, then verify Home, Single Prediction, Batch Audit, both downloads, and
   Model Evidence using the built-in sample.

The app processes uploads in memory and does not write them to disk. Streamlit
Community Cloud infrastructure and logs remain governed by Streamlit's service terms;
do not upload real sensitive or regulated data to this classroom deployment.

## Tests

```bash
pytest -q
```

## Generative-AI acknowledgement

AI-assisted tools were used to support code scaffolding, documentation, and testing. The student team remains responsible for understanding, validating, explaining, and presenting the work. No generative AI should be used to substantially produce the required video presentation unless the instructor explicitly permits it.

## Submission checklist

- [ ] Replace smoke-test artifacts with a full fine-tuned model.
- [ ] Run `pytest -q` with all tests passing.
- [ ] Run held-out evaluation and retain `artifacts/metrics.json`.
- [ ] Demonstrate error handling with an invalid CSV.
- [ ] Record a clear video presentation (mandatory under the syllabus).
- [ ] Acknowledge AI assistance and cite data/model sources.
- [ ] Confirm requirements against the separate project-details file.
