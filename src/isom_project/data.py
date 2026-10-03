"""Synthetic, privacy-preserving data for the classroom demonstration.

The text contains neutral proxy tokens rather than real names, addresses, or
demographic identities. The group label exists only to measure audit error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, SEED


GROUP_A_TOKENS = ["alden", "brin", "corva", "delmar", "elwin", "faron"]
GROUP_B_TOKENS = ["navi", "oriel", "pavan", "quinn", "ruma", "soren"]
SHARED_TOKENS = ["marlow", "reese", "taylor", "vale"]
REGIONS = ["north", "south", "east", "west"]
CHANNELS = ["mobile", "branch", "web", "partner"]
PRODUCTS = ["credit", "savings", "insurance", "mortgage"]


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def generate_dataset(n: int = 6000, seed: int = SEED) -> pd.DataFrame:
    """Generate an overlapping two-group proxy-text audit dataset."""
    if n < 100:
        raise ValueError("n must be at least 100 so every split is usable")

    rng = np.random.default_rng(seed)
    group_b = rng.binomial(1, 0.5, n)
    region_idx = np.where(
        group_b == 1,
        rng.choice(4, n, p=[0.15, 0.40, 0.30, 0.15]),
        rng.choice(4, n, p=[0.38, 0.18, 0.18, 0.26]),
    )
    region = np.asarray(REGIONS)[region_idx]
    channel = rng.choice(CHANNELS, n)
    product = rng.choice(PRODUCTS, n)

    proxy_tokens: list[str] = []
    for label in group_b:
        # 25% overlap prevents the text task from becoming an identity oracle.
        if rng.random() < 0.25:
            proxy_tokens.append(rng.choice(SHARED_TOKENS))
        else:
            pool = GROUP_B_TOKENS if label else GROUP_A_TOKENS
            proxy_tokens.append(rng.choice(pool))

    income_band = rng.integers(1, 6, n)
    tenure_years = rng.integers(0, 16, n)
    qualified_probability = _sigmoid(
        -0.7 + 0.42 * income_band + 0.055 * tenure_years
        + 0.18 * (product == "savings")
    )
    qualified = rng.binomial(1, qualified_probability)

    # A direct group penalty creates a disparity for the downstream audit.
    decision_probability = _sigmoid(
        -1.0 + 1.9 * qualified + 0.08 * tenure_years
        - 0.65 * group_b + 0.18 * (channel == "branch")
    )
    decision = rng.binomial(1, decision_probability)

    text = [
        f"profile token {token}; region {reg}; channel {chan}; product {prod}; "
        f"income band {inc}; tenure {tenure} years"
        for token, reg, chan, prod, inc, tenure in zip(
            proxy_tokens, region, channel, product, income_band, tenure_years
        )
    ]

    order = rng.permutation(n)
    split = np.full(n, "train", dtype=object)
    split[order[int(n * 0.70):int(n * 0.85)]] = "validation"
    split[order[int(n * 0.85):]] = "test"

    return pd.DataFrame(
        {
            "record_id": [f"SYN-{i:06d}" for i in range(n)],
            "text": text,
            "group_b": group_b.astype(int),
            "qualified": qualified.astype(int),
            "decision": decision.astype(int),
            "split": split,
        }
    )


def write_dataset(output_dir: Path = DATA_DIR, n: int = 6000, seed: int = SEED) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "synthetic_audit_data.csv"
    generate_dataset(n=n, seed=seed).to_csv(path, index=False)
    return path


def sample_audit_data(n: int = 250, seed: int = SEED + 7) -> pd.DataFrame:
    return generate_dataset(max(n, 100), seed=seed).head(n).copy()

