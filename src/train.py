"""Train the FinAI 360 fraud module (demonstration).

Pipeline: synthetic transactions -> XGBoost classifier (class-weighted for
the 98/2 imbalance) with an Isolation Forest anomaly baseline, threshold
tuning for the approve/review/block operating point, and SHAP explanation.

Saves the fitted pipeline and chosen threshold to ``models/`` so the
scoring API (``scoring_api.py``) can serve it. All data is synthetic.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
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


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Pick the threshold maximizing F1, and print the trade-off table."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best_idx = int(np.argmax(f1))
    print("\nThreshold trade-off (precision / recall / F1):")
    for t in (0.3, 0.5, 0.7, 0.9):
        idx = int(np.searchsorted(thresholds, t))
        idx = min(idx, len(precision) - 1)
        print(f"  threshold {t:.1f}: precision {precision[idx]:.3f}, "
              f"recall {recall[idx]:.3f}, F1 {f1[idx]:.3f}")
    best = float(thresholds[min(best_idx, len(thresholds) - 1)])
    print(f"Recommended operating threshold (max F1): {best:.3f}")
    return best


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
    xgb_score = xgb.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_score >= 0.5).astype(int)
    print("\n===== XGBoost (supervised) =====")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, xgb_pred))
    print(classification_report(y_test, xgb_pred, digits=3))
    print(f"ROC-AUC: {roc_auc_score(y_test, xgb_score):.4f}")
    print(f"PR-AUC (avg precision): {average_precision_score(y_test, xgb_score):.4f}")

    # --- Unsupervised baseline: Isolation Forest ------------------------
    iso = IsolationForest(
        n_estimators=200, contamination=float(y_train.mean()),
        random_state=42, n_jobs=-1
    )
    iso.fit(X_train[NUMERIC])
    iso_score = -iso.decision_function(X_test[NUMERIC])
    iso_pred = (iso.predict(X_test[NUMERIC]) == -1).astype(int)
    print("\n===== Isolation Forest (unsupervised baseline) =====")
    print(f"ROC-AUC: {roc_auc_score(y_test, iso_score):.4f}")
    print(f"PR-AUC: {average_precision_score(y_test, iso_score):.4f}")

    # --- Threshold tuning + SHAP -----------------------------------------
    threshold = tune_threshold(y_test, xgb_score)
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
        print("\nSHAP not installed; skipping explanation.")
    except Exception as exc:
        print(f"\nSHAP explanation skipped ({exc}).")

    # --- Save artifacts for the scoring API ------------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb, "models/fraud_pipeline.pkl")
    with open("models/threshold.json", "w") as f:
        json.dump({"review_threshold": 0.5, "block_threshold": 0.8,
                   "tuned_threshold": threshold}, f, indent=2)
    print("\nSaved models/fraud_pipeline.pkl and models/threshold.json")
    print("Serve with: python src/scoring_api.py")


if __name__ == "__main__":
    main()
