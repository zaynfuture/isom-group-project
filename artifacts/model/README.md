---
pipeline_tag: text-classification
library_name: transformers
base_model: google/bert_uncased_L-2_H-128_A-2
---

# FairnessLens synthetic demographic imputer

This classroom model was fine-tuned on synthetic Group A/Group B proxy text for
testing downstream fairness-estimation error. It is not validated for real
demographic inference and must not be used for individual decisions.

## Held-out test metrics

```json
{
  "test_loss": 0.2713862657546997,
  "test_accuracy": 0.8922222222222222,
  "test_macro_precision": 0.8995553841156407,
  "test_macro_recall": 0.8894448869342383,
  "test_macro_f1": 0.8910589741573918,
  "test_roc_auc": 0.9766967284888122,
  "test_brier_score": 0.0725875198841095,
  "test_runtime": 0.1475,
  "test_samples_per_second": 6100.786,
  "test_steps_per_second": 196.581,
  "epoch": 4.0
}
```
