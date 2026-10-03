---
library_name: chronos
license: apache-2.0
pipeline_tag: time-series-forecasting
base_model: amazon/chronos-bolt-tiny
---
# SpendLens spending forecast — Chronos-Bolt Tiny

Fine-tuned on synthetic demo; not a production-validated financial model. No demographic inference.

Input: two complete monthly positive purchase totals per card, in USD. Output: next-month p10/p50/p90 and derived LOW/MEDIUM/HIGH bands. Labels thresholds: 200 and 400 USD.

Best epoch: 3, selected on validation pinball loss. Test MAE: 89.6436; test band Macro-F1: 0.3756; baseline MAE: 92.1452; baseline band Macro-F1: 0.3864.

Only two monthly observations per input. Demo future amounts have little predictable structure. Same cards recur across temporal splits; July test targets were previously inspected for DistilBERT. Real external validation is required. No demographic inference. Quantile coverage is empirical, not guaranteed.

Install `chronos-forecasting==2.3.2` and `transformers==4.57.6`. Load with `BaseChronosPipeline.from_pretrained(repo_id, device_map='cpu')`; call `predict(torch.tensor([[250., 320.]]), prediction_length=1)`. Output shape is batch × quantiles × horizon. See forecast_contract.json and evaluation.json.

Training script: https://github.com/zaynfuture/isom-group-project/blob/main/scripts/train_spendlens_chronos.py
