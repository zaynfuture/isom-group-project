"""Public, deterministic data partitions shared by visible notebook experiments."""
import numpy as np
from model.inference.forecasting import numeric_examples

SEED = 5240

def _cards(frame, smoke):
    cards = np.array(sorted(frame.card_id.unique()))
    np.random.default_rng(SEED).shuffle(cards)
    return cards[:12] if smoke else cards

def merchant_split(frame, smoke=False):
    # Refunds copy purchase descriptions, so never make them extra training examples.
    frame = frame.loc[frame.amount.gt(0)].copy()
    cards = _cards(frame, smoke)
    if len(cards) < 10:
        raise ValueError('At least 10 cards are required.')
    n = len(cards)
    groups = {'train':cards[:int(n*.7)], 'validation':cards[int(n*.7):int(n*.85)], 'test':cards[int(n*.85):]}
    return {key:frame.loc[frame.card_id.isin(ids)].reset_index(drop=True) for key,ids in groups.items()}

def forecast_split(frame, smoke=False):
    if set(frame.currency) != {'USD'}:
        raise ValueError('Forecast experiments require USD; normalize explicitly first.')
    examples = numeric_examples(frame.loc[frame.card_id.isin(_cards(frame,smoke))])
    if examples.empty or examples.target_month.nunique()<3:
        raise ValueError('At least six complete calendar months are required.')
    months = sorted(examples.target_month.unique())
    return {'train':examples.loc[examples.target_month < months[-2]].reset_index(drop=True),
            'validation':examples.loc[examples.target_month == months[-2]].reset_index(drop=True),
            'test':examples.loc[examples.target_month == months[-1]].reset_index(drop=True)}
