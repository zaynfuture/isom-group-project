"""Numeric monthly spending interface for Chronos-Bolt; no identity features."""
import numpy as np
import pandas as pd

BASE_FORECAST_MODEL = "amazon/chronos-bolt-tiny"
BANDS = ["HIGH", "LOW", "MEDIUM"]


def spending_bands(amounts):
    values = np.asarray(amounts)
    return np.where(values < 200, "LOW", np.where(values < 400, "MEDIUM", "HIGH"))


def numeric_examples(frame):
    """Backtest: two prior monthly purchase totals predict one month.

    Same targets/history length as the earlier text classifier. Last observed
    month excluded because completeness is unknown. Complete account coverage
    is a caller requirement; zero monthly activity is zero, not missing data.
    """
    if frame.empty or set(frame.currency) != {"USD"}:
        raise ValueError("Forecasting requires nonempty USD transactions.")
    data = frame.assign(month=frame.timestamp.dt.tz_localize(None).dt.to_period("M"))
    months = pd.period_range(data.month.min(), data.month.max(), freq="M")
    rows = []
    for card, part in data.groupby("card_id"):
        totals = part.loc[part.amount.gt(0)].groupby("month").amount.sum().reindex(months, fill_value=0)
        for i in range(2, len(months)-1):
            context = totals.iloc[i-2:i].to_numpy(float).tolist()
            target = float(totals.iloc[i])
            rows.append({"card_id":card, "target_month":str(months[i]), "context":context,
                         "target":target, "label":spending_bands([target])[0], "baseline_amount":float(np.mean(context))})
    return pd.DataFrame(rows)


def predict_amounts(pipe, contexts):
    import torch
    rows = []
    for offset in range(0, len(contexts), 32):
        batch = torch.tensor(contexts[offset:offset+32], dtype=torch.float32)
        raw = pipe.predict(batch, prediction_length=1).numpy()[:, :, 0]
        if not np.isfinite(raw).all():
            raise ValueError("Forecast contains nonfinite values.")
        # Enforce nonnegative purchases and noncrossing quantiles consistently.
        ordered = np.sort(np.maximum(raw, 0), axis=1)
        levels = np.asarray(pipe.quantiles)
        for values in ordered:
            rows.append({"p10":float(np.interp(.1, levels, values)),
                         "median_amount":float(np.interp(.5, levels, values)),
                         "p90":float(np.interp(.9, levels, values))})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["prediction"] = spending_bands(result.median_amount)
    return result
