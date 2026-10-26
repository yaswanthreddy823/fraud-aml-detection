"""Streaming fraud scoring demo (simulated Kafka stream).

Production consumed real Kafka topics; this demo simulates the stream with
an in-memory generator so the pipeline runs anywhere. Each transaction is
scored by a small XGBoost model, passed through the business-rule layer,
and decided as approve / review / block. A summary shows how the layered
decision performs against the synthetic ground truth.

Run:  python src/streaming_demo.py [--n 1000]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from generate_data import make_dataset
from rules import combine_decision, evaluate_rules

NUMERIC = [
    "amount",
    "hour",
    "account_age_days",
    "distance_km",
    "txns_last_24h",
    "is_new_device",
    "is_international",
]
CATEGORICAL = ["merchant_category", "device_type"]


def train_small_model() -> Pipeline:
    """Fit a compact model on a training sample for the demo stream."""
    df = make_dataset(n=10_000, seed=1)
    X = df[NUMERIC + CATEGORICAL]
    y = df["is_fraud"].to_numpy()
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    neg, pos = int((y == 0).sum()), int((y == 1).sum())
    pipe = Pipeline(
        [
            ("prep", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    scale_pos_weight=neg / max(pos, 1),
                    eval_metric="logloss",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000,
                        help="transactions to stream")
    args = parser.parse_args()

    print("Training small scoring model...")
    pipe = train_small_model()
    stream = make_dataset(n=args.n, seed=99)
    flagged, labels, shown = [], [], 0
    for _, row in stream.iterrows():
        txn = row.to_dict()
        score = float(pipe.predict_proba(
            row[NUMERIC + CATEGORICAL].to_frame().T)[0, 1])
        flags = evaluate_rules(txn)
        decision = combine_decision(score, flags)
        flagged.append(1 if decision != "approve" else 0)
        labels.append(int(row["is_fraud"]))
        if decision != "approve" and shown < 5:
            print(f"ALERT {decision:>6} score={score:.3f} flags={flags} "
                  f"amount=${row['amount']:.0f} fraud={int(row['is_fraud'])}")
            shown += 1

    print(f"\nStreamed {args.n} transactions, "
          f"flagged {sum(flagged)} for review/block.")
    print(classification_report(labels, flagged, digits=3,
                                target_names=["legit", "flagged"]))


if __name__ == "__main__":
    main()
