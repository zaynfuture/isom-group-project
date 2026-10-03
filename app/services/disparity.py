"""Card-level exploratory disparity screening; not demographic inference."""
import numpy as np
import pandas as pd

FIELDS = ("gender", "age_band", "ethnicity")
METRICS = ("Purchase amount per card", "Purchase count per card", "Category purchase share")


def demo_labels(card_ids):
    """Independent synthetic labels, never predictions of identity."""
    cards = sorted(set(card_ids))
    rng = np.random.default_rng(725)
    return pd.DataFrame({
        "card_id": cards,
        "gender": rng.permutation(np.resize(["Woman", "Man", "Non-binary"], len(cards))),
        "age_band": rng.permutation(np.resize(["18–34", "35–54", "55+"], len(cards))),
        "ethnicity": rng.permutation(np.resize(["Synthetic group A", "Synthetic group B", "Synthetic group C"], len(cards))),
        "consent": True, "source": "Independent synthetic demo labels",
    })


def validate_labels(labels):
    if not {"card_id", "consent", "source", *FIELDS}.issubset(labels.columns):
        raise ValueError("Labels require card_id, gender, age_band, ethnicity, consent and source.")
    labels = labels.copy()
    for col in ["card_id", "source"]:
        labels[col] = labels[col].fillna("").astype(str).str.strip()
        if labels[col].eq("").any():
            raise ValueError(f"{col} cannot be blank.")
    if labels.card_id.duplicated().any():
        raise ValueError("Provide exactly one label row per card; duplicate cards are not allowed.")
    if not labels.consent.astype(str).str.strip().str.lower().isin(["true", "1"]).all():
        raise ValueError("All supplied labels must have consent=true.")
    for field in FIELDS:
        labels[field] = labels[field].fillna("").astype(str).str.strip()
        labels.loc[labels[field].str.lower().isin(["", "unknown", "prefer not to say", "not disclosed"]), field] = "Unknown"
    return labels


def card_metrics(transactions, metric, category=None):
    if transactions.empty or transactions.currency.nunique() != 1:
        raise ValueError("Select nonempty transactions in one currency.")
    if metric not in METRICS:
        raise ValueError("Unsupported metric.")
    cards = pd.Index(sorted(transactions.card_id.unique()), name="card_id")
    purchases = transactions.loc[transactions.amount.gt(0)]
    totals = purchases.groupby("card_id").amount.sum().reindex(cards, fill_value=0)
    if metric == METRICS[0]:
        values = totals
    elif metric == METRICS[1]:
        values = purchases.groupby("card_id").size().reindex(cards, fill_value=0)
    else:
        if category not in set(transactions.category):
            raise ValueError("Choose an available category.")
        numerator = purchases.loc[purchases.category.eq(category)].groupby("card_id").amount.sum().reindex(cards, fill_value=0)
        values = numerator.div(totals.where(totals.gt(0)))
    return values.rename("value").reset_index()


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def screen(values, labels, field, reference, minimum_cards=10, practical_gap=0.0, draws=999):
    """Resample independent cards, not repeated transactions. Holm per report."""
    if field not in FIELDS or minimum_cards < 10 or draws < 99:
        raise ValueError("Use a supported field, at least 10 cards and at least 99 resamples.")
    if not np.isfinite(practical_gap) or practical_gap < 0:
        raise ValueError("Practical gap must be finite and nonnegative.")
    labels = validate_labels(labels)
    joined = values.merge(labels[["card_id", field]], on="card_id", how="left", validate="one_to_one")
    joined[field] = joined[field].fillna("Unknown")
    valid = joined.loc[np.isfinite(joined.value) & joined[field].ne("Unknown")]
    groups = {str(k): g.value.to_numpy(float) for k, g in valid.groupby(field) if len(g) >= minimum_cards}
    coverage = {"observed_cards": len(joined), "known_label_cards": int(joined[field].ne("Unknown").sum()),
                "missing_label_cards": int(joined[field].eq("Unknown").sum()),
                "undefined_metric_cards": int((~np.isfinite(joined.value)).sum()),
                "suppressed_groups": int((valid.groupby(field).size() < minimum_cards).sum())}
    if reference not in groups or len(groups) < 2:
        raise ValueError("Insufficient data: choose a reference and at least one other group with the minimum card count.")
    if len(groups) > 30 or len(joined) > 5000:
        raise ValueError("Interactive screening supports at most 30 eligible groups and 5,000 observed cards.")
    ref = groups[reference]
    rows = []
    rng = np.random.default_rng(5240)
    for name, x in groups.items():
        if name == reference:
            continue
        gap = float(x.mean() - ref.mean())
        pool = np.concatenate([x, ref])
        exceed = 0
        boot = np.empty(draws)
        for i in range(draws):
            perm = rng.permutation(pool)
            exceed += abs(perm[:len(x)].mean() - perm[len(x):].mean()) >= abs(gap) - 1e-12
            boot[i] = rng.choice(x, len(x), replace=True).mean() - rng.choice(ref, len(ref), replace=True).mean()
        lo, hi = np.quantile(boot, [.025, .975])
        rows.append({"group": name, "cards": len(x), "mean": x.mean(), "reference": reference,
                     "reference_cards": len(ref), "reference_mean": ref.mean(), "gap": gap,
                     "mean_ratio": x.mean()/ref.mean() if ref.mean() != 0 else np.nan,
                     "gap_ci_low": lo, "gap_ci_high": hi, "p_value": (1+exceed)/(draws+1)})
    result = pd.DataFrame(rows)
    result["p_holm"] = holm(result.p_value)
    result["review_flag"] = result.p_holm.lt(.05) & result.gap.abs().gt(practical_gap)
    result["assessment"] = np.where(result.review_flag, "Observed disparity — review context", "No flag under selected settings; not proof of fairness")
    return result, coverage
