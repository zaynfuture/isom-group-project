# SpendLens implementation contract

## Business objective
Fictional Meridian Card Services supports portfolio analysts with category coverage, spending forecasts and authorized cohort comparisons. Predict behavior, not demographic identity. Card identifiers are anonymized tokens; cards are not necessarily people.

## Two model pipelines
1. Merchant description -> text normalization/tokenization -> pretrained DistilBERT fine-tuned for five demo category labels -> probabilities and review flag.
2. Two complete months of category purchase totals -> temporal text -> separately fine-tuned pretrained DistilBERT -> next-month USD purchase band.

Base model: distilbert/distilbert-base-uncased. Fine-tuning remains a later explicit Colab action. Random classification heads are never served as trained models. Existing FairnessLens results do not transfer.

## Data contract
Required CSV fields: transaction_id, card_id, timestamp, amount, currency, mcc, merchant_description, city, channel. Unique transaction IDs, finite signed amounts, parseable UTC timestamps. Refunds are negative. MCC is four digits or missing. Five-code demonstration crosswalk; all other MCCs stay unmapped. Separate currencies; no assumed FX conversion. Maximum 100,000 records per batch. Demo: 7,680 transactions, 120 cards, eight months.

Labels: optional card_id, age_band/gender/ethnicity, consent=true, source. No demographic inference. Missing labels remain Unknown. Suppress cohort/category cells below 10 cards; this is not a formal privacy guarantee.

## Evaluation
Merchant: disjoint card splits, 70/15/15, majority baseline, accuracy, Macro-F1, confusion matrix. Shared merchants may limit external generalization.
Forecast: chronological target-month splits, last evaluable month test, preceding month validation. Earlier months train. Last observed month excluded as a target; incomplete monthly coverage must be resolved upstream. Baseline uses previous two-month average. Fixed demo USD bands: <200, <400, >=400. Compare Macro-F1; tune thresholds with business owners before real adoption.

## Deployment and acceptance
App files -> existing public GitHub repository -> existing Streamlit Cloud URL. Model files -> zhengzhihust/spendlens-merchant-classifier and zhengzhihust/spendlens-spending-forecast. Upload model, tokenizer, model card, training history and evaluation evidence, not raw transaction CSVs. Configure SPENDLENS_MERCHANT_MODEL and SPENDLENS_FORECAST_MODEL only after artifacts exist. No new training/upload performed automatically.

Acceptance: full demo browse, currency-separated summaries, strict input validation, no future target data in model inputs, honest pending-model UI. ML completion additionally requires both fine-tuning runs, external review, Hub uploads and cloud inference verification. Report/PPT/video remain separate deliverables.

## Practical readiness boundary
Public synthetic demo and code suitable for development/testing. Real financial records require a controlled deployment with authentication, access controls, retention policy and an approved data source. Production readiness is not claimed.

## References
- Primary MCC crosswalk source provided by user: https://www.citibank.com/tts/solutions/commercial-cards/assets/docs/govt/Merchant-Category-Codes.pdf (reviewed codes 5311, 5411, 5541, 5812, 7011; normalized descriptions; mapping version 2026-10-03).
- https://usa.visa.com/dam/VCOM/download/merchants/visa-merchant-data-standards-manual.pdf
- https://huggingface.co/distilbert/distilbert-base-uncased
