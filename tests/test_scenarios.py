import pandas as pd
from model.data.scenarios import scenario_transactions, VERSION
from model.data.transactions import demo_transactions, validate_transactions

def test_coverage_reproducibility_and_activity():
    data = scenario_transactions()
    pd.testing.assert_frame_equal(data,scenario_transactions())
    assert len(data)>500
    assert data.category.nunique()==14
    assert data.groupby('synthetic_lifestyle').card_id.nunique().eq(20).all()
    assert data.groupby('card_id').size().nunique()>20
    assert data.groupby('category').size().min()>50
    assert set(data.scenario_version)=={VERSION}
    assert data.transaction_id.is_unique
    assert data.amount.ne(0).all()
    pd.testing.assert_frame_equal(data,validate_transactions(data))

def test_refunds_are_linked_bounded_and_later():
    data=scenario_transactions()
    purchases=data.loc[data.amount.gt(0)].set_index('transaction_id')
    refunds=data.loc[data.amount.lt(0)]
    assert .01 < len(refunds)/len(purchases) < .15
    assert set(refunds.transaction_type)=={'full_refund','partial_refund'}
    original=purchases.loc[refunds.original_transaction_id].reset_index()
    refunds=refunds.reset_index(drop=True)
    assert refunds.card_id.eq(original.card_id).all()
    assert refunds.mcc.eq(original.mcc).all()
    assert refunds.currency.eq(original.currency).all()
    assert refunds.amount.abs().le(original.amount).all()
    assert (refunds.timestamp-original.timestamp).dt.days.between(2,21).all()
    assert refunds.timestamp.dt.month.ne(original.timestamp.dt.month).any()
    assert refunds.loc[refunds.transaction_type.eq('full_refund'),'amount'].abs().eq(original.loc[refunds.transaction_type.eq('full_refund'),'amount']).all()
    assert refunds.original_transaction_id.is_unique

def test_recurring_and_legacy_fixture_is_preserved():
    data=scenario_transactions()
    bills=data.loc[data.recurring]
    assert bills.groupby(['card_id','mcc']).amount.nunique().eq(1).all()
    assert bills.groupby(['card_id','mcc']).timestamp.nunique().eq(8).all()
    old=demo_transactions()
    assert len(old)==7680 and old.category.nunique()==5 and old.amount.gt(0).all()
