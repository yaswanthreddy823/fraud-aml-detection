"""Real-time scoring API for the fraud module (demonstration).

Mirrors the production pattern: a Python ML API that Java Spring Boot
microservices call to score transactions in real time. The API loads the
pipeline trained by ``train.py`` and applies the business-rule layer from
``rules.py`` before returning a decision.

Run ``python src/train.py`` first to create ``models/`` artifacts, then::

    python src/scoring_api.py

    curl -X POST localhost:5000/score -H 'Content-Type: application/json' -d '{
      "amount": 1250.0, "hour": 3, "merchant_category": "electronics",
      "device_type": "mobile", "account_age_days": 45, "distance_km": 820.0,
      "txns_last_24h": 7, "is_new_device": 1, "is_international": 1}'
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from rules import combine_decision, evaluate_rules

app = Flask(__name__)

REQUIRED_FIELDS = [
    "amount",
    "hour",
    "merchant_category",
    "device_type",
    "account_age_days",
    "distance_km",
    "txns_last_24h",
    "is_new_device",
    "is_international",
]


def load_artifacts():
    if not os.path.exists("models/fraud_pipeline.pkl"):
        raise FileNotFoundError(
            "models/fraud_pipeline.pkl not found. Run 'python src/train.py' first."
        )
    pipeline = joblib.load("models/fraud_pipeline.pkl")
    with open("models/threshold.json") as f:
        thresholds = json.load(f)
    return pipeline, thresholds


PIPELINE, THRESHOLDS = load_artifacts()


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/score")
def score():
    txn = request.get_json(force=True, silent=True) or {}
    missing = [f for f in REQUIRED_FIELDS if f not in txn]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400
    row = pd.DataFrame([{f: txn[f] for f in REQUIRED_FIELDS}])
    risk_score = float(PIPELINE.predict_proba(row)[0, 1])
    flags = evaluate_rules(txn)
    decision = combine_decision(
        risk_score, flags,
        review_threshold=THRESHOLDS.get("review_threshold", 0.5),
        block_threshold=THRESHOLDS.get("block_threshold", 0.8),
    )
    return jsonify({
        "risk_score": round(risk_score, 4),
        "decision": decision,
        "rules_triggered": flags,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
