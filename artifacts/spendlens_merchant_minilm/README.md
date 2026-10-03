---
library_name: transformers
license: mit
pipeline_tag: text-classification
base_model: microsoft/MiniLM-L12-H384-uncased
---
# SpendLens merchant classifier — MiniLM

Task-specific fine-tuned model. Labels: ['Department stores', 'Dining', 'Fuel', 'Grocery', 'Lodging'].

Training data source: synthetic demo. No demographic identity inference. Evaluation is in evaluation.json.

Fine-tuned for three epochs from Microsoft MiniLM (33,361,925 parameters). Best checkpoint: first epoch, selected on validation Macro-F1. Card-held-out split sizes: 5,376 training / 1,152 validation / 1,152 test. Test accuracy and Macro-F1: 1.000; majority baseline Macro-F1: 0.06897. The prior DistilBERT also scored 1.000: the improvement here is model footprint, not demonstrated classification accuracy.

The data contain only five category-specific merchant-description templates reused across splits. A perfect score is therefore a template-recognition demonstration, not evidence of generalization to real or unseen merchants. The test data were inspected in prior experiments. Validate on representative, independently collected descriptions before use. Softmax scores are uncalibrated. Known MCC mappings remain authoritative; model output is only a suggestion. No demographic identity inference or automated customer eligibility decisions.

```python
from transformers import pipeline
model = pipeline('text-classification', model='zhengzhihust/spendlens-merchant-minilm', top_k=None)
print(model(['Fresh Market 12', 'Harbor Hotel 8']))
```

Reproducible training: https://github.com/zaynfuture/isom-group-project/blob/main/SpendLens_Colab.ipynb
Official base model: https://huggingface.co/microsoft/MiniLM-L12-H384-uncased (MIT).
