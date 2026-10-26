"""Synthetic transaction data generator for fraud detection experiments.

Generates realistic-looking card transactions with a small fraud minority.
Fraud is driven by interpretable risk signals (unusual amount, odd hour,
new device, distant location, high velocity) plus noise, so models must
learn signal rather than memorize a rule.

All data is synthetic. No real customer or employer data is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MERCHANT_CATEGORIES = [
    "grocery",
    "electronics",
    "travel",
    "dining",
    "fuel",
    "online_retail",
    "entertainment",
]
DEVICE_TYPES = ["mobile", "desktop", "pos", "tablet"]


def make_dataset(
    n: int = 50_000, fraud_rate: float = 0.02, seed: int = 42
) -> pd.DataFrame:
    """Return a DataFrame of synthetic transactions with an ``is_fraud`` label."""
    rng = np.random.default_rng(seed)

    amount = rng.lognormal(mean=3.5, sigma=1.1, size=n)  # median ~$33
    hour = rng.integers(0, 24, size=n)
    merchant = rng.choice(
        MERCHANT_CATEGORIES,
        size=n,
        p=[0.28, 0.12, 0.08, 0.18, 0.10, 0.16, 0.08],
    )
    device = rng.choice(DEVICE_TYPES, size=n, p=[0.45, 0.25, 0.20, 0.10])
    account_age_days = rng.integers(30, 3650, size=n)
    distance_km = np.abs(rng.normal(loc=15.0, scale=40.0, size=n))
    txns_last_24h = rng.poisson(lam=2.0, size=n)
    is_new_device = (rng.random(size=n) < 0.15).astype(int)
    is_international = (rng.random(size=n) < 0.12).astype(int)

    # Latent fraud risk: risk signals plus noise.
    logit = (
        -4.0
        + 0.9 * (amount > 500)
        + 0.7 * (hour < 5)
        + 0.8 * is_new_device
        + 0.6 * (distance_km > 200)
        + 0.5 * is_international
        + 0.4 * (txns_last_24h >= 6)
        + 0.3 * (account_age_days < 180)
        + rng.normal(0, 1.0, size=n)
    )
    prob = 1 / (1 + np.exp(-logit))
    # Threshold on ranked risk so the fraud share matches fraud_rate exactly.
    threshold = np.quantile(prob, 1 - fraud_rate)
    is_fraud = (prob >= threshold).astype(int)

    return pd.DataFrame(
        {
            "amount": np.round(amount, 2),
            "hour": hour,
            "merchant_category": merchant,
            "device_type": device,
            "account_age_days": account_age_days,
            "distance_km": np.round(distance_km, 1),
            "txns_last_24h": txns_last_24h,
            "is_new_device": is_new_device,
            "is_international": is_international,
            "is_fraud": is_fraud,
        }
    )


if __name__ == "__main__":
    df = make_dataset()
    out = "data/transactions.csv"
    import os

    os.makedirs("data", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out} "
          f"({df['is_fraud'].mean():.1%} fraud)")
