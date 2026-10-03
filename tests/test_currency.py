import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datetime import date
import pandas as pd
import pytest
from app.services.currency import FXBook, FXError, convert_transactions, display_forecast


def book():
    return FXBook({"source":"test fixture (not live FX)","retrieved_at":"2026-01-01T00:00:00Z", "records":[
        {"base":"EUR","quote":"USD","date":"2025-01-03","rate":"1.2"},
        {"base":"EUR","quote":"SGD","date":"2025-01-03","rate":"1.8"},
        {"base":"EUR","quote":"JPY","date":"2025-01-03","rate":"150"},
        {"base":"EUR","quote":"USD","date":"2025-01-06","rate":"1.5"},
        {"base":"EUR","quote":"SGD","date":"2025-01-06","rate":"3"}]})


def transactions():
    return pd.DataFrame({"card_id":["a","b","c"],"amount":[120.,180.,-150.],
                         "currency":["USD","SGD","JPY"],"timestamp":pd.to_datetime(["2025-01-04"]*3,utc=True)})


def test_cross_rates_refund_identity_and_input_preserved():
    raw = transactions()
    result = convert_transactions(raw,"USD",book())
    assert result.amount.tolist() == pytest.approx([120,120,-1.2])
    assert result.currency.eq("USD").all()
    assert result.original_amount.tolist() == raw.amount.tolist()
    assert raw.currency.tolist() == ["USD","SGD","JPY"]
    assert result.iloc[1].fx_source_date == "2025-01-03"
    assert convert_transactions(raw.iloc[:1],"USD").fx_rate.iloc[0] == 1
    assert float(book().quote("USD","EUR",date(2025,1,4))[0]) == pytest.approx(1/1.2)


def test_never_use_future_or_stale_or_missing_rates():
    rates = book()
    assert float(rates.quote("USD","SGD",date(2025,1,4))[0]) == 1.5
    with pytest.raises(FXError,match="No historical"):
        rates.quote("USD","SGD",date(2025,1,2))
    with pytest.raises(FXError,match="older"):
        rates.quote("USD","SGD",date(2025,2,1))
    with pytest.raises(FXError):
        rates.quote("USD","XXX",date(2025,1,4))
    with pytest.raises(FXError,match="snapshot"):
        convert_transactions(transactions(),"USD")


def test_bad_rates_fail_closed():
    for rate in ["0","-1","NaN","Infinity"]:
        with pytest.raises(FXError,match="positive"):
            FXBook({"source":"test","retrieved_at":"test","records":[{"base":"EUR","quote":"USD","date":"2025-01-01","rate":rate}]})


def test_forecast_currency_does_not_relabel_usd_bands():
    rates = FXBook({"source":"test","retrieved_at":"test","records":[
        {"base":"EUR","quote":"USD","date":"2025-01-31","rate":"1.2"},
        {"base":"EUR","quote":"SGD","date":"2025-01-31","rate":"1.8"},
        {"base":"EUR","quote":"SGD","date":"2025-02-01","rate":"9"}]})
    result = pd.DataFrame({"p10":[100.],"median_amount":[300.],"p90":[500.],"prediction":["MEDIUM"]})
    display = display_forecast(result,["2025-02"],"SGD",rates)
    assert display.median_amount.iloc[0] == 450
    assert display.median_amount_usd.iloc[0] == 300
    assert display.prediction.iloc[0] == "MEDIUM"
    assert display.display_fx_date.iloc[0] == "2025-01-31"
