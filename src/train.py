"""Train and evaluate fraud detection models on synthetic transactions.

Models:
  1. XGBoost classifier (supervised, class-weighted for 98/2 imbalance).
  2. Isolation Forest (unsupervised anomaly baseline).

Metrics focus on precision-recall, which matters more than accuracy when
fraud is rare. SHAP explanation runs when the ``shap`` package is installed
and is skipped gracefully otherwise.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from generate_data import make_dataset

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


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray,
           y_score: np.ndarray) -> None:
    print(f"\n===== {name} =====")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=3))
    print(f"ROC-AUC: {roc_auc_score(y_true, y_score):.4f}")
    print(f"PR-AUC (avg precision): {average_precision_score(y_true, y_score):.4f}")


def main() -> None:
    df = make_dataset()
    X = df[NUMERIC + CATEGORICAL]
    y = df["is_fraud"].to_numpy()
    print(f"Dataset: {len(df):,} rows, {y.mean():.1%} fraud")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )

    # --- Supervised: XGBoost -------------------------------------------
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    xgb = Pipeline(
        [
            ("prep", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=neg / max(pos, 1),
                    eval_metric="logloss",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_score = xgb.predict_proba(X_test)[:, 1]
    report("XGBoost (supervised)", y_test, xgb_pred, xgb_score)

    # --- Unsupervised baseline: Isolation Forest ------------------------
    iso = IsolationForest(
        n_estimators=200, contamination=float(y_train.mean()),
        random_state=42, n_jobs=-1
    )
    iso.fit(X_train[NUMERIC])
    iso_score = -iso.decision_function(X_test[NUMERIC])
    iso_pred = (iso.predict(X_test[NUMERIC]) == -1).astype(int)
    report("Isolation Forest (unsupervised)", y_test, iso_pred, iso_score)

    # --- Explainability: SHAP -------------------------------------------
    try:
        import shap

        model = xgb.named_steps["model"]
        prep = xgb.named_steps["prep"]
        X_test_t = prep.transform(X_test)
        feature_names = prep.get_feature_names_out()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_t)
        importance = np.abs(shap_values).mean(axis=0)
        order = np.argsort(importance)[::-1][:10]
        print("\nTop 10 features by mean |SHAP value|:")
        for i in order:
            print(f"  {feature_names[i]:<35} {importance[i]:.4f}")
    except ImportError:
        print("\nSHAP not installed; skipping explanation "
              "(pip install shap to enable it).")
    except Exception as exc:  # keep the script runnable without SHAP
        print(f"\nSHAP explanation skipped ({exc}).")


if __name__ == "__main__":
    main()
