import pytest
from model.data.transactions import demo_transactions
from model.experiments import merchant_split, forecast_split

def test_merchant_card_partitions_are_disjoint_and_deterministic():
    frame = demo_transactions()
    parts = merchant_split(frame, smoke=True)
    assert set(parts)=={'train','validation','test'}
    assert parts['train'].card_id.nunique()==8
    assert parts['validation'].card_id.nunique()==2
    assert parts['test'].card_id.nunique()==2
    assert not set(parts['train'].card_id) & set(parts['test'].card_id)
    assert not set(parts['train'].card_id) & set(parts['validation'].card_id)
    assert not set(parts['validation'].card_id) & set(parts['test'].card_id)
    assert parts['test'].equals(merchant_split(frame,smoke=True)['test'])

def test_forecast_uses_temporal_partitions_and_rejects_mixed_currency():
    frame=demo_transactions()
    parts=forecast_split(frame,smoke=True)
    assert set(parts['train'].target_month)=={'2025-03','2025-04','2025-05'}
    assert set(parts['validation'].target_month)=={'2025-06'}
    assert set(parts['test'].target_month)=={'2025-07'}
    frame.loc[0,'currency']='SGD'
    with pytest.raises(ValueError,match='USD'):
        forecast_split(frame)
