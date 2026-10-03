import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pandas as pd
import pytest
from isom_project.spending import demo_transactions, validate_transactions, monthly_examples, labeled_cohorts


def test_demo_and_mapping_roundtrip():
    data = demo_transactions()
    assert len(data) == 7680
    assert data.card_id.nunique() == 120
    assert data.category.nunique() == 5
    assert validate_transactions(data).amount.sum() == data.amount.sum()


def test_duplicates_rejected():
    data = demo_transactions().iloc[:2].copy()
    data["transaction_id"] = "duplicate"
    with pytest.raises(ValueError,match="unique"):
        validate_transactions(data)


def test_forecast_inputs_exclude_target_month():
    data = demo_transactions()
    before = monthly_examples(data)
    changed = data.copy()
    changed.loc[changed.timestamp.dt.month.eq(7),"amount"] = 9999
    after = monthly_examples(changed)
    assert before.loc[before.target_month.eq("2025-07"),"text"].tolist() == after.loc[after.target_month.eq("2025-07"),"text"].tolist()
    assert set(before.target_month) == {"2025-03","2025-04","2025-05","2025-06","2025-07"}


def test_cohort_consent_and_small_cell_suppression():
    data = demo_transactions()
    labels = pd.DataFrame({"card_id":["CARD-0000"],"gender":["voluntary label"],"consent":[False],"source":["survey"]})
    with pytest.raises(ValueError,match="consent"):
        labeled_cohorts(data,labels,"gender")
    labels["consent"] = True
    summary = labeled_cohorts(data,labels,"gender")
    assert "voluntary label" not in set(summary.gender)
