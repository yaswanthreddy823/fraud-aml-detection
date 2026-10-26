"""Synthetic transaction data generator for fraud detection experiments.

Two profiles generate the data: legitimate transactions and a small fraud
minority (~2%). Fraud rows are drawn from shifted distributions on
observable features (larger amounts, odd hours, new devices, far
locations, high velocity, young accounts), so the label is learnable from
the features with realistic overlap -- the way fraud behaves in the real
world -- rather than from an unobservable noise term.

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

LEGIT_MERCHANT_P = [0.28, 0.12, 0.08, 0.18, 0.10, 0.16, 0.08]
FRAUD_MERCHANT_P = [0.20, 0.16, 0.10, 0.14, 0.08, 0.20, 0.12]
LEGIT_DEVICE_P = [0.45, 0.25, 0.20, 0.10]
FRAUD_DEVICE_P = [0.48, 0.27, 0.15, 0.10]
# Daytime-weighted hours for legitimate traffic; fraud skews to odd hours.
LEGIT_HOUR_P = np.array(
    [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 0.07, 0.07, 0.07, 0.07,
     0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01]
)
LEGIT_HOUR_P /= LEGIT_HOUR_P.sum()
FRAUD_HOUR_P = np.array(
    [0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
     0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]
)
FRAUD_HOUR_P /= FRAUD_HOUR_P.sum()


def make_dataset(
    n: int = 50_000, fraud_rate: float = 0.02, seed: int = 42
) -> pd.DataFrame:
    """Return a DataFrame of synthetic transactions with an ``is_fraud`` label."""
    rng = np.random.default_rng(seed)
    is_fraud = (rng.random(n) < fraud_rate).astype(int)
    fraud = is_fraud == 1

    # Baseline (legitimate) distributions for every row.
    amount = rng.lognormal(mean=3.5, sigma=1.1, size=n)
    hour = rng.choice(24, size=n, p=LEGIT_HOUR_P)
    merchant = rng.choice(MERCHANT_CATEGORIES, size=n, p=LEGIT_MERCHANT_P)
    device = rng.choice(DEVICE_TYPES, size=n, p=LEGIT_DEVICE_P)
    account_age_days = rng.integers(30, 3650, size=n)
    distance_km = np.abs(rng.normal(loc=15.0, scale=40.0, size=n))
    txns_last_24h = rng.poisson(lam=2.0, size=n)
    is_new_device = (rng.random(size=n) < 0.15).astype(int)
    is_international = (rng.random(size=n) < 0.12).astype(int)

    # Fraud profile: shift the observable distributions for fraud rows.
    m = int(fraud.sum())
    amount[fraud] = rng.lognormal(mean=4.5, sigma=1.2, size=m)
    hour[fraud] = rng.choice(24, size=m, p=FRAUD_HOUR_P)
    merchant[fraud] = rng.choice(MERCHANT_CATEGORIES, size=m, p=FRAUD_MERCHANT_P)
    device[fraud] = rng.choice(DEVICE_TYPES, size=m, p=FRAUD_DEVICE_P)
    account_age_days[fraud] = rng.integers(30, 2400, size=m)
    distance_km[fraud] = np.abs(rng.normal(loc=100.0, scale=200.0, size=m))
    txns_last_24h[fraud] = rng.poisson(lam=3.0, size=m)
    is_new_device[fraud] = (rng.random(size=m) < 0.35).astype(int)
    is_international[fraud] = (rng.random(size=m) < 0.25).astype(int)

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
