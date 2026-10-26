"""Smoke tests for the fraud module: data, rules, decisions.

Run:  python -m pytest tests/ -q
No model training here, so the suite stays fast.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generate_data import make_dataset
from rules import combine_decision, evaluate_rules

NORMAL_TXN = {
    "amount": 45.0,
    "hour": 14,
    "merchant_category": "grocery",
    "device_type": "pos",
    "account_age_days": 900,
    "distance_km": 5.0,
    "txns_last_24h": 1,
    "is_new_device": 0,
    "is_international": 0,
}

FRAUD_LIKE_TXN = {
    **NORMAL_TXN,
    "amount": 2400.0,
    "hour": 3,
    "distance_km": 900.0,
    "txns_last_24h": 9,
    "is_new_device": 1,
    "is_international": 1,
}


def test_dataset_shape_and_fraud_rate():
    df = make_dataset(n=5000, seed=7)
    assert len(df) == 5000
    assert "is_fraud" in df.columns
    assert 0.015 <= df["is_fraud"].mean() <= 0.025


def test_rules_fire_on_fraud_like_transaction():
    flags = evaluate_rules(FRAUD_LIKE_TXN)
    assert "high_amount" in flags
    assert "odd_hour" in flags
    assert "velocity" in flags


def test_rules_quiet_on_normal_transaction():
    assert evaluate_rules(NORMAL_TXN) == []


def test_decision_escalation():
    assert combine_decision(0.95, []) == "block"
    assert combine_decision(0.10, []) == "approve"
    # Two rule flags escalate a low score to review.
    assert combine_decision(0.10, ["velocity", "odd_hour"]) == "review"
