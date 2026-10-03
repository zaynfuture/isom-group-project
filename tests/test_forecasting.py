import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from isom_project.forecasting import numeric_examples, spending_bands, predict_amounts
from isom_project.spending import demo_transactions, monthly_examples


def test_numeric_targets_match_original_and_exclude_future():
    data = demo_transactions()
    numeric = numeric_examples(data)
    original = monthly_examples(data)
    assert len(numeric) == 600
    assert np.allclose(numeric.target, original.future_purchase_amount)
    changed = data.copy()
    changed.loc[changed.timestamp.dt.month.eq(7),"amount"] *= 10
    later = numeric_examples(changed)
    assert numeric.loc[numeric.target_month.eq("2025-07"),"context"].tolist() == later.loc[later.target_month.eq("2025-07"),"context"].tolist()
    assert set(numeric.target_month) == {"2025-03","2025-04","2025-05","2025-06","2025-07"}


def test_bands_and_quantile_postprocessing():
    import torch
    assert spending_bands([0,199.99,200,399.99,400]).tolist() == ["LOW","LOW","MEDIUM","MEDIUM","HIGH"]
    class Fake:
        quantiles = [.1,.5,.9]
        def predict(self, batch, prediction_length):
            return torch.tensor([[[400.],[250.],[-10.]]]*len(batch))
    result = predict_amounts(Fake(), [[100,200],[200,300]])
    assert result.p10.eq(0).all()
    assert result.median_amount.eq(250).all()
    assert result.p90.eq(400).all()
    assert result.prediction.eq("MEDIUM").all()
