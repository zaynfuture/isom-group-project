"""SpendLens transaction contracts, deterministic demo and temporal examples."""
import numpy as np
import pandas as pd
from .mcc_reference import REFERENCE, VERSION, SPENDING_REFERENCE, SPENDING_VERSION

# Deliberately limited demonstration crosswalk, not the complete MCC standard.
MCC = {code: item["category"] for code, item in REFERENCE.items()}
MERCHANTS = {"5411": "Fresh Market", "5812": "Garden Restaurant", "5541": "City Fuel", "5311": "Central Department Store", "7011": "Harbor Hotel"}
REQUIRED = {"transaction_id", "card_id", "timestamp", "amount", "currency", "mcc", "merchant_description", "city", "channel"}


def demo_transactions():
    rng = np.random.default_rng(5240)
    rows = []
    for card in range(120):
        for month in range(1, 9):
            for j in range(8):
                code = rng.choice(list(MCC))
                rows.append({"transaction_id": f"T-{card:04}-{month:02}-{j}", "card_id": f"CARD-{card:04}",
                    "timestamp": f"2025-{month:02}-{int(rng.integers(1,28)):02}T12:00:00Z",
                    "amount": round(float(rng.lognormal(3.4, .7)), 2), "currency": "USD", "mcc": code,
                    "merchant_description": f"{MERCHANTS[code]} {int(rng.integers(1,50))}",
                    "city": rng.choice(["New York", "Boston", "Chicago"]), "channel": rng.choice(["online", "in_store"])})
    return validate_transactions(pd.DataFrame(rows))


def validate_transactions(frame):
    missing = REQUIRED - set(frame)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    if frame.empty or len(frame) > 100_000:
        raise ValueError("Supply between 1 and 100,000 transactions per batch.")
    result = frame.copy()
    for name in REQUIRED - {"amount", "timestamp", "mcc"}:
        if result[name].isna().any() or result[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} contains missing or blank values.")
        result[name] = result[name].astype(str).str.strip()
    if result.transaction_id.duplicated().any():
        raise ValueError("transaction_id must be unique; remove duplicate imports.")
    result["timestamp"] = pd.to_datetime(result.timestamp, utc=True, errors="raise")
    if result.timestamp.isna().any():
        raise ValueError("timestamp cannot be missing.")
    result["amount"] = pd.to_numeric(result.amount, errors="raise")
    if not np.isfinite(result.amount).all():
        raise ValueError("Amounts must be finite.")
    result["currency"] = result.currency.str.upper()
    if not result.currency.str.fullmatch(r"[A-Z]{3}").all():
        raise ValueError("Use three-letter currency codes.")
    result["mcc"] = result.mcc.fillna("").astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    if not (result.mcc.eq("") | result.mcc.str.fullmatch(r"\d{4}")).all():
        raise ValueError("MCC must be four digits or blank.")
    result["category"] = result.mcc.map({code: item["category"] for code,item in SPENDING_REFERENCE.items()}).fillna("Other / unmapped")
    result["mcc_description"] = result.mcc.map({code: item["description"] for code,item in SPENDING_REFERENCE.items()}).fillna("Not in reviewed reference subset")
    result["mapping_version"] = SPENDING_VERSION
    return result.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)


def monthly_examples(frame):
    """Two complete calendar months of positive purchases predict the next month.

    Last observed month is excluded as a target because completeness is unknown.
    Callers must supply complete account coverage; zero activity is treated as zero.
    """
    if frame.currency.nunique() != 1:
        raise ValueError("Select one currency before building forecasting examples.")
    data = frame.assign(month=frame.timestamp.dt.tz_localize(None).dt.to_period("M"))
    periods = pd.period_range(data.month.min(), data.month.max(), freq="M")
    categories = list(MCC.values())
    rows = []
    for card, part in data.groupby("card_id"):
        for i in range(2, len(periods)-1):
            context = part.loc[part.month.isin(periods[i-2:i]) & part.amount.gt(0)]
            pieces = []
            for month in periods[i-2:i]:
                monthly = context.loc[context.month.eq(month)]
                shares = monthly.groupby("category").amount.sum()
                pieces.append(str(month) + " " + " ".join(f"{cat}: {shares.get(cat,0):.2f}" for cat in categories))
            future = part.loc[part.month.eq(periods[i]) & part.amount.gt(0), "amount"].sum()
            label = "LOW" if future < 200 else "MEDIUM" if future < 400 else "HIGH"
            rows.append({"card_id": card, "text": " ; ".join(pieces), "label": label,
                         "target_month": str(periods[i]), "future_purchase_amount": float(future),
                         "baseline_label": "LOW" if context.amount.sum()/2 < 200 else "MEDIUM" if context.amount.sum()/2 < 400 else "HIGH"})
    return pd.DataFrame(rows)


def labeled_cohorts(transactions, labels, field, minimum_cards=10):
    if field not in {"age_band", "gender", "ethnicity"}:
        raise ValueError("Unsupported demographic field.")
    if not {"card_id", field, "consent", "source"}.issubset(labels):
        raise ValueError("Labels require card_id, selected field, consent and source.")
    if labels.card_id.duplicated().any():
        raise ValueError("Provide at most one label row per card.")
    if not labels.consent.astype(str).str.lower().isin(["true", "1"]).all():
        raise ValueError("All label records must have consent=true.")
    if labels.source.fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Document the label source for every record.")
    labels = labels.copy()
    labels[field] = labels[field].fillna("Unknown").replace("", "Unknown")
    joined = transactions.merge(labels[["card_id", field]], on="card_id", how="left", validate="many_to_one")
    joined[field] = joined[field].fillna("Unknown")
    summary = joined.groupby([field, "category"]).agg(cards=("card_id", "nunique"), transactions=("amount", "size"), net_amount=("amount", "sum")).reset_index()
    return summary.loc[summary.cards.ge(minimum_cards)]
