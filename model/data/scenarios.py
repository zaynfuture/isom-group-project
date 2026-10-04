"""Versioned behavioral simulation, not an empirical population model.

Demographics are never inputs. USD amounts represent partial card-wallet activity,
not all household spending. Legacy training fixtures remain unchanged.
"""
import calendar
import numpy as np
import pandas as pd
from .transactions import validate_transactions

VERSION = "lifestyle-demo-v2-seed5240"
# MCC: merchant, typical ticket, relative frequency, refund probability, online share
SCENARIOS = {
    "5411": ("Neighborhood Market", 45, 20, .02, .20),
    "5812": ("Corner Kitchen", 28, 18, .015, .25),
    "5541": ("Roadside Fuel", 48, 9, .01, .0),
    "5311": ("Central Store", 70, 6, .12, .50),
    "7011": ("Riverside Stay", 230, 2, .08, .90),
    "4111": ("Metro Transit", 6, 15, .01, .10),
    "4511": ("Regional Air", 320, 1, .08, .95),
    "4814": ("Connect Mobile", 38, 4, .01, 1.),
    "4900": ("City Utilities", 95, 4, .01, 1.),
    "5651": ("Everyday Apparel", 65, 6, .18, .65),
    "5732": ("Home Electronics", 210, 2, .12, .70),
    "5912": ("Community Pharmacy", 22, 6, .02, .20),
    "7832": ("Neighborhood Cinema", 20, 5, .05, .70),
    "7997": ("Active Sports Club", 45, 2, .02, 1.),
}
PROFILES = {
    "Transit & city dining": {"4111":3., "5812":2., "5541":.15},
    "Home & essentials": {"5411":2.5, "4900":2., "5912":1.8},
    "Driving & errands": {"5541":3., "5311":1.6, "4111":.15},
    "Travel & leisure": {"4511":5., "7011":5., "7832":2.},
    "Digital shopping": {"5651":2.5, "5732":3., "5311":2.},
    "Active & local": {"7997":4., "5411":1.5, "7832":1.7},
}

def scenario_transactions():
    rng = np.random.default_rng(5240)
    rows = []
    codes = list(SCENARIOS)
    end = pd.Timestamp("2025-08-31T23:59:59Z")
    for card in range(120):
        profile = list(PROFILES)[card % len(PROFILES)]
        scale = float(rng.uniform(.65, 1.5))
        activity = [.45, .85, 1.3][(card // 6) % 3]
        home = str(rng.choice(["New York", "Boston", "Chicago"]))
        weights = np.array([SCENARIOS[c][2]*PROFILES[profile].get(c,1.) for c in codes],dtype=float)
        weights /= weights.sum()
        billing_day = int(rng.integers(2, 9))
        for month in range(1,9):
            # Variable account activity, with stable recurring bills for subscribers.
            selected = list(rng.choice(codes,size=max(3,int(rng.poisson(16*activity))),p=weights))
            selected = [c for c in selected if c not in {"4814","4900","7997"}]
            if card % 2 == 0: selected.append("4814")
            if card % 3 == 0: selected.append("4900")
            if profile == "Active & local": selected.append("7997")
            for code in selected:
                merchant, ticket, _, refund_rate, online = SCENARIOS[code]
                recurring = code in {"4814","4900","7997"}
                days = np.arange(1,calendar.monthrange(2025,month)[1]+1)
                weekdays = np.array([pd.Timestamp(2025,month,int(d)).dayofweek for d in days])
                day_weights = np.ones(len(days))
                if code == "4111": day_weights[weekdays < 5] = 3.
                if code in {"7832","5812","5651","5311"}: day_weights[weekdays >= 5] = 2.
                day = billing_day if recurring else int(rng.choice(days,p=day_weights/day_weights.sum()))
                hour = int(rng.choice([7,8,9,17,18])) if code == "4111" else int(rng.integers(10,22))
                timestamp = pd.Timestamp(year=2025,month=month,day=day,hour=hour,tz="UTC")
                amount = round(float(ticket*scale*(1 if recurring else rng.lognormal(0,.5))),2)
                purchase = {"transaction_id":f"S-{len(rows):06d}","card_id":f"CARD-{card:04d}",
                    "timestamp":timestamp,"amount":amount,"currency":"USD","mcc":code,
                    "merchant_description":f"{merchant} {card % 12 + 1}",
                    "city":str(rng.choice(["New York","Boston","Chicago"])) if code in {"4511","7011"} else home,
                    "channel":"online" if rng.random()<online else "in_store",
                    "transaction_type":"purchase","original_transaction_id":"",
                    "refund_reason":"","synthetic_lifestyle":profile,"recurring":recurring,
                    "scenario_version":VERSION}
                rows.append(purchase)
                if rng.random() < refund_rate:
                    refund_date = timestamp + pd.Timedelta(days=int(rng.integers(2,22)))
                    if refund_date <= end:
                        full = bool(rng.random() < .6)
                        refund = purchase.copy()
                        refund.update(transaction_id=f"S-{len(rows):06d}",timestamp=refund_date,
                            amount=-round(amount*(1. if full else float(rng.uniform(.2,.8))),2),
                            transaction_type="full_refund" if full else "partial_refund",
                            original_transaction_id=purchase["transaction_id"],recurring=False,
                            refund_reason="Cancellation" if code in {"4511","7011","7832"} else "Return or billing adjustment")
                        rows.append(refund)
    return validate_transactions(pd.DataFrame(rows))
