# Product Specification: FairnessLens Web Application

## Purpose

FairnessLens helps Responsible AI teams determine whether probabilistically imputed demographic attributes are reliable enough for aggregate fairness screening when verified attributes are unavailable. It is an educational decision-support tool, not an identity classifier or legal-compliance engine.

## Users

- Data scientists and ML engineers validating model behavior.
- Responsible AI and model-risk reviewers.
- Researchers demonstrating demographic-imputation uncertainty.

## Primary user journey

1. Read the intended-use boundary and acknowledge aggregate-audit use.
2. Select the built-in demonstration data or upload a CSV.
3. Validate schema, binary fields, missing text, sample size, and optional benchmark labels.
4. Run batched demographic-probability inference.
5. Compare soft-imputed fairness metrics with observed metrics when benchmark labels exist.
6. Interpret metric direction, magnitude, uncertainty, and limitations.
7. Download scored records and a reproducible audit summary.

## Functional requirements

- FR-001: The interface and generated report must be entirely in English.
- FR-002: The application must require an intended-use acknowledgement before inference.
- FR-003: A batch must include `text`, `qualified`, and `decision`; each binary column must contain only 0/1.
- FR-004: `group_b` is optional and, when present, must contain only 0/1.
- FR-005: Inference must retain `P(Group B)` and use it for soft-imputed metrics.
- FR-006: The audit must report selection rates, demographic-parity gap, disparate-impact ratio, equal-opportunity gap, and false-positive-rate gap.
- FR-007: When `group_b` exists, the audit must show observed, imputed, and absolute-error rows.
- FR-008: Users must be able to download scored CSV data and a JSON audit summary.
- FR-009: Model evaluation must distinguish demographic classification quality from downstream fairness-estimation validity.
- FR-010: The app must load a bundled model or a Hugging Face model selected through `HF_MODEL_ID`.
- FR-011: The application must run on Streamlit Community Cloud without local filesystem assumptions beyond repository artifacts.
- FR-012: Users must be able to inspect all 6,000 synthetic records and filter the scrollable table by training, validation, or testing split.

## Non-functional requirements

- Cached model loading; batched inference; deterministic sample data.
- No uploaded data persistence by application code.
- Helpful validation errors without stack traces in the UI.
- Responsive wide layout and accessible text labels.
- Explicit educational-use and prohibited-use statements on every workflow boundary.

## Acceptance criteria

- The built-in sample contains at least 500 records and completes the end-to-end journey with both downloads.
- Invalid or missing columns prevent inference and identify the exact problem.
- With benchmark labels, the app displays fairness-estimation error.
- Without benchmark labels, the app labels the output as screening-only.
- Automated tests cover input validation, metric direction, and report construction.
- A clean environment can run `streamlit run app.py` using `requirements.txt`.

## Course alignment extension

- Business case: Meridian Financial Services, explicitly fictional; Responsible AI analysts review aggregate credit-decision disparities.
- Retrospective project targets: Macro-F1 >= .85, ROC AUC >= .90, Brier <= .10, DP-gap absolute error <= .10; never describe these as preregistered criteria.
- Pipeline 1: Hugging Face pretrained BERT, subsequently fine-tuned offline for demographic probability inference.
- Pipeline 2: validated probability-weighted fairness aggregation, benchmark comparison and report export. This is statistical processing, not a second ML model.
- The course slide does not define whether two application workflows suffice or two Hugging Face task types are required; instructor interpretation remains unconfirmed.
- Model Evidence displays recorded training loss, per-epoch validation scores, checkpoint selection, saved-model split evaluation and test confusion matrix.
- Existing results are historical evidence. Future fine-tuning remains an explicit Colab step; app startup never trains a model.
- Test labels are excluded from gradient updates and checkpoint selection, but results were inspected across iterations; no claim of single-use external validation.

## Out of scope

- Inferring a real person's race, ethnicity, gender, age, or other identity.
- Individual eligibility, pricing, hiring, lending, or enforcement decisions.
- A legal determination of compliance or fairness.
- Authentication, long-term data storage, or production monitoring.
