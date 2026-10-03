"""Auditable historical FX: Frankfurter rates + CurrencyConverter arithmetic.

Only currency codes and date ranges leave the application. Monetary amounts,
transaction rows and customer identifiers never go to the rate provider.
"""
from bisect import bisect_right
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from currency_converter import CurrencyConverter
from babel.numbers import get_currency_name, format_currency

API = "https://api.frankfurter.dev/v2"
DEFAULT_CURRENCY = "USD"
COMMON_CURRENCIES = "USD EUR GBP CNY JPY SGD HKD AUD CAD CHF NZD KRW INR IDR MYR THB PHP TWD VND AED SAR QAR KWD BHD OMR ZAR BRL MXN SEK NOK DKK PLN CZK HUF TRY ILS CLP COP PEN ARS EGP PKR BDT NGN KES UAH RUB RON ISK MAD".split()
SOURCE = "Frankfurter v2 — blended reference rates"


class FXError(ValueError):
    """No defensible conversion can be made; fail closed."""


def currency_code(value):
    value = str(value).strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", value):
        raise FXError("Use three-letter ISO currency codes.")
    return value


def currency_label(code):
    return f"{code} — {get_currency_name(code, locale='en_US')}"


def money(value, code):
    return format_currency(value, code, locale="en_US") + f" {code}"


def _get(endpoint, params=None):
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=.3, status_forcelist=[429,500,502,503,504], allowed_methods=["GET"])))
        try:
            response = session.get(API + endpoint, params=params, timeout=(5,20))
            response.raise_for_status()
            return response.json(parse_float=str)
        except (requests.RequestException, ValueError) as exc:
            raise FXError("Exchange-rate service is unavailable or returned invalid data. Retry later; no rates were substituted.") from exc


def fetch_catalog():
    rows = _get("/currencies")
    if not isinstance(rows,list):
        raise FXError("Invalid currency catalogue.")
    try:
        return sorted({currency_code(row["iso_code"]) for row in rows})
    except (KeyError,TypeError) as exc:
        raise FXError("Malformed currency catalogue.") from exc


def fetch_snapshot(currencies, start, end):
    codes = sorted({currency_code(c) for c in currencies} - {"EUR"})
    if end < start or (end-start).days > 3660 or end > datetime.now(timezone.utc).date():
        raise FXError("FX windows must be historical, ordered and no longer than ten years.")
    rows = _get("/rates", {"base":"EUR", "quotes":",".join(codes),
                           "from":str(start-timedelta(days=7)), "to":str(end)}) if codes else []
    if not isinstance(rows,list) or len(rows) > 200_000:
        raise FXError("Invalid or oversized rate response.")
    return {"source":SOURCE,"retrieved_at":datetime.now(timezone.utc).isoformat(),"records":rows}


class FXBook:
    def __init__(self, snapshot, max_age_days=7):
        self.source = snapshot["source"]
        self.retrieved_at = snapshot["retrieved_at"]
        self.snapshot_id = hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()[:16]
        self.max_age_days = max_age_days
        self.rates = {}
        for row in snapshot["records"]:
            if row.get("base") != "EUR":
                raise FXError("Expected EUR-referenced rates.")
            try:
                quote = currency_code(row["quote"])
                observed = date.fromisoformat(row["date"])
                rate = Decimal(str(row["rate"]))
            except (KeyError,TypeError,ValueError,InvalidOperation) as exc:
                raise FXError("Malformed historical rate record.") from exc
            if not rate.is_finite() or rate <= 0:
                raise FXError("Rates must be finite and positive.")
            series = self.rates.setdefault(quote,{})
            if observed in series and series[observed] != rate:
                raise FXError("Conflicting rates for the same date and currency.")
            series[observed] = rate
        self.dates = {code:sorted(series) for code,series in self.rates.items()}
        self._cache = {}

    def _asof(self, code, requested):
        if code == "EUR":
            return Decimal(1),requested
        days = self.dates.get(code,[])
        index = bisect_right(days,requested)-1
        if index < 0:
            raise FXError(f"No historical {code} rate on or before {requested}.")
        actual = days[index]
        if (requested-actual).days > self.max_age_days:
            raise FXError(f"{code} rate is older than {self.max_age_days} days on {requested}.")
        return self.rates[code][actual],actual

    def quote(self, base, target, requested):
        base,target = currency_code(base),currency_code(target)
        if base == target:
            return Decimal(1),requested,requested
        key = base,target,requested
        if key not in self._cache:
            source_rate,source_date = self._asof(base,requested)
            target_rate,target_date = self._asof(target,requested)
            rates = {base:source_rate,target:target_rate}
            rates.pop("EUR",None)
            converter = CurrencyConverter(currency_file=None,decimal=True)
            converter.load_lines(["Date,"+",".join(rates),str(requested)+","+",".join(str(r) for r in rates.values())])
            factor = converter.convert(Decimal(1),base,target,date=requested)
            self._cache[key] = factor,source_date,target_date
        return self._cache[key]


def convert_transactions(frame, target="USD", book=None):
    """Keep originals and per-row provenance; do not round before aggregation."""
    target = currency_code(target)
    if frame.empty:
        raise FXError("No transactions to convert.")
    result = frame.copy()
    result["original_amount"] = result.amount
    result["original_currency"] = result.currency
    converted,meta = [],[]
    for row in result.itertuples():
        day = row.timestamp.date()
        if row.currency == target:
            factor,source_date,target_date = Decimal(1),day,day
            provider,retrieved,snapshot = "Identity (no FX)","","identity"
        else:
            if book is None:
                raise FXError("A verified FX snapshot is required for cross-currency conversion.")
            factor,source_date,target_date = book.quote(row.currency,target,day)
            provider,retrieved,snapshot = book.source,book.retrieved_at,book.snapshot_id
        value = Decimal(str(row.amount)) * factor
        if not value.is_finite():
            raise FXError("Amount conversion produced a nonfinite value.")
        converted.append(float(value))
        meta.append({"fx_rate":float(factor),"fx_source_date":str(source_date),"fx_target_date":str(target_date),
                     "fx_provider":provider,"fx_retrieved_at":retrieved,"fx_snapshot_id":snapshot})
    if not np.isfinite(converted).all():
        raise FXError("Converted values exceed the supported numeric range.")
    result["amount"],result["currency"] = converted,target
    for column in meta[0]:
        result[column] = [entry[column] for entry in meta]
    return result


def display_forecast(result, target_months, currency, book=None):
    """Keep USD outputs and bands. Display conversion uses forecast-origin FX."""
    if len(result) != len(target_months):
        raise FXError("Forecast rows and target months must align.")
    converted = result.copy()
    for column in ["p10","median_amount","p90"]:
        converted[column+"_usd"] = converted[column]
    rate_dates = []
    for index,month in zip(converted.index,target_months):
        cutoff = (pd.Period(month,freq="M").start_time - pd.Timedelta(days=1)).date()
        if currency == "USD":
            rate,observed = Decimal(1),cutoff
        elif book is not None:
            rate,a,b = book.quote("USD",currency,cutoff)
            observed = min(a,b)
        else:
            raise FXError("An FX snapshot is required for forecast display.")
        for column in ["p10","median_amount","p90"]:
            converted.loc[index,column] = float(Decimal(str(converted.loc[index,column+"_usd"]))*rate)
        rate_dates.append(str(observed))
    converted["display_currency"] = currency
    converted["display_fx_date"] = rate_dates
    return converted
