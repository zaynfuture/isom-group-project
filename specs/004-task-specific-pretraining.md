# Task-appropriate pretrained models

## Decision
Replace the shared text-model assumption with separate model families:
- Merchant descriptions: microsoft/MiniLM-L12-H384-uncased (MIT, approximately 33M parameters). Lower model footprint than DistilBERT; downstream accuracy must be measured, not inferred from size.
- Next-month amount: amazon/chronos-bolt-tiny (Apache-2.0, approximately 9M parameters). Numeric time-series pretraining and direct quantile predictions match the target better than text classification of serialized numbers. Two historical months remain a severe information limit.
- FairnessLens remains authorized-label statistical screening, not an identity inference model.

Official sources checked on 2026-10-03:
https://huggingface.co/microsoft/MiniLM-L12-H384-uncased
https://huggingface.co/amazon/chronos-bolt-tiny
https://github.com/amazon-science/chronos-forecasting

Alternatives reviewed: Chronos-T5 Tiny is small but autoregressive sampling adds inference overhead; Chronos-2 is newer and supports more inputs but exceeds this small univariate prototype's needs. Selection is based on task fit and deployment footprint, not a claim of global state of the art.

## Data, selection and evaluation
Preserve old DistilBERT artifacts. Never overwrite them with a candidate. Same synthetic transaction generator and target months are retained. Merchant splits hold out cards, but the five merchant templates overlap across splits: perfect scores are not evidence of unseen-merchant generalization. MiniLM: three epochs, select on validation Macro-F1. Chronos: five epochs, AdamW 1e-4, batch 32, select on validation mean pinball loss. Test does not select epochs. The test set was inspected during prior DistilBERT work and is not a new external holdout.

Chronos uses two numeric total-purchase observations per card rather than category text, with the same target months. Chronological split: March–May training, June validation, July test; August excluded as potentially incomplete. Report MAE, band Macro-F1, accuracy, pinball loss and empirical p10–p90 coverage, plus prior-two-month baseline. No demographic fields enter either model.

## Publication and serving
New repositories under authorized zhengzhihust account: spendlens-merchant-minilm and spendlens-spending-chronos-bolt-tiny. Only allowlisted model/config/tokenizer/evaluation/card files are uploaded. No raw transactions, identity labels or credentials. Require local save/reload verification, then remote pinned-revision reload and prediction agreement. Serving manifest pins commit SHA and matching evidence directory. Keep weights out of Git. Colab reproduces scripts and interactive login; Streamlit never trains on startup.

## Local evidence gap
Local knowledge health checked. Search observation search-93694006-7075-40c2-92af-dc3a6c3380ab, 462 estimated tokens within 1,500 cap. Passages passage-0c3479e6394c05545eec53992cef8dbd and passage-fac5288bec69000ef59572c7db96f5d9 were unrelated and not used for model selection. Local corpus lacked relevant Hugging Face model documentation; external official model cards were consulted. Context truncation reported by retrieval.
