"""
evaluation.py
-------------
Evaluates trained models on the held-out test split.
Generates comprehensive comparative metrics, threshold sensitivity analysis,
business cost calculations, and feature importance rankings.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    FEATURE_IMPORTANCE_FILE,
    FEATURE_LIST_FILE,
    ISO_MODEL_FILE,
    LOG_FILE,
    LOG_LEVEL,
    LOGISTIC_MODEL_FILE,
    MODEL_COMPARISON_FILE,
    OPTIMAL_CLASSIFICATION_THRESHOLD,
    REPORTS_DIR,
    RF_MODEL_FILE,
    TEST_FILE,
    THRESHOLD_CONFIG_FILE,
    XGB_MODEL_FILE,
)
from src.feature_engineering import engineer_features
from src.preprocessing import get_feature_names, load_pipeline, transform
from src.anomaly_detection import get_anomaly_flags, get_anomaly_scores, load_isolation_forest

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger("evaluation")


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = OPTIMAL_CLASSIFICATION_THRESHOLD,
    model_name: str = "",
) -> Dict:
    """Calculate complete fraud analytics metric suite."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # Financial cost assumption: Average fraud loss = $1,500; False alert investigation cost = $35
    fraud_loss_cost = fn * 1500
    investigation_cost = fp * 35
    total_business_cost = fraud_loss_cost + investigation_cost

    return {
        "model": model_name,
        "threshold": threshold,
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positive_rate": round(float(fp / (fp + tn) if (fp + tn) > 0 else 0), 4),
        "false_negative_rate": round(float(fn / (fn + tp) if (fn + tp) > 0 else 0), 4),
        "flagged_transactions": int(y_pred.sum()),
        "estimated_business_cost_usd": int(total_business_cost),
    }


def perform_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "XGBoost",
) -> pd.DataFrame:
    """Analyze precision, recall, and cost trade-offs across 10 probability thresholds."""
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    results = [compute_metrics(y_true, y_prob, threshold=t, model_name=model_name) for t in thresholds]
    df = pd.DataFrame(results)

    # Save threshold analysis report
    threshold_path = REPORTS_DIR / "threshold_analysis.csv"
    df.to_csv(threshold_path, index=False)
    logger.info("Saved threshold sensitivity report to %s", threshold_path)

    # Save optimal threshold configuration
    best_row = df.loc[df["f1"].idxmax()]
    optimal_cfg = {
        "optimal_threshold": float(best_row["threshold"]),
        "f1_at_optimal": float(best_row["f1"]),
        "precision_at_optimal": float(best_row["precision"]),
        "recall_at_optimal": float(best_row["recall"]),
        "recommended_risk_bands": {
            "low": [0.0, 0.30],
            "medium": [0.30, 0.70],
            "high": [0.70, 1.0],
        }
    }
    with open(THRESHOLD_CONFIG_FILE, "w") as f:
        json.dump(optimal_cfg, f, indent=2)
    logger.info("Saved optimal threshold configuration to %s", THRESHOLD_CONFIG_FILE)

    return df


def extract_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """Extract and rank feature importances for tree-based models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        df = pd.DataFrame({
            "feature": feature_names[:len(importances)],
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        df.to_csv(FEATURE_IMPORTANCE_FILE, index=False)
        logger.info("Saved feature importance ranking to %s (Top feature: %s)",
                    FEATURE_IMPORTANCE_FILE, df.iloc[0]["feature"])
        return df
    return pd.DataFrame()


def evaluate_test_set() -> pd.DataFrame:
    """
    Run full test suite:
    1. Load test data & transform
    2. Load trained Logistic Regression, Random Forest, XGBoost, and Isolation Forest
    3. Generate side-by-side comparison table
    4. Run threshold analysis on primary model
    5. Output final comparison CSV
    """
    logger.info("Loading test dataset from %s ...", TEST_FILE)
    test_df = pd.read_csv(TEST_FILE, low_memory=False)
    test_fe = engineer_features(test_df)

    pipeline = load_pipeline()
    X_test, y_test = transform(pipeline, test_fe)
    feature_names = get_feature_names()

    comparison_rows = []

    # 1. Logistic Regression
    if LOGISTIC_MODEL_FILE.exists():
        lr = joblib.load(LOGISTIC_MODEL_FILE)
        lr_probs = lr.predict_proba(X_test)[:, 1]
        comparison_rows.append(compute_metrics(y_test, lr_probs, threshold=0.5, model_name="Logistic Regression"))

    # 2. Random Forest
    if RF_MODEL_FILE.exists():
        rf = joblib.load(RF_MODEL_FILE)
        rf_probs = rf.predict_proba(X_test)[:, 1]
        comparison_rows.append(compute_metrics(y_test, rf_probs, threshold=0.5, model_name="Random Forest"))

    # 3. XGBoost
    if XGB_MODEL_FILE.exists():
        xgb_mod = joblib.load(XGB_MODEL_FILE)
        xgb_probs = xgb_mod.predict_proba(X_test)[:, 1]
        comparison_rows.append(compute_metrics(y_test, xgb_probs, threshold=OPTIMAL_CLASSIFICATION_THRESHOLD, model_name="XGBoost"))
        # Run threshold optimization and feature importance
        perform_threshold_analysis(y_test, xgb_probs, model_name="XGBoost")
        extract_feature_importance(xgb_mod, feature_names)

    # 4. Isolation Forest (Unsupervised benchmark)
    if ISO_MODEL_FILE.exists():
        iso = load_isolation_forest()
        iso_scores = get_anomaly_scores(iso, X_test)
        iso_flags = get_anomaly_flags(iso, X_test)
        cm_iso = confusion_matrix(y_test, iso_flags)
        tn_i, fp_i, fn_i, tp_i = cm_iso.ravel() if cm_iso.shape == (2, 2) else (0, 0, 0, 0)
        comparison_rows.append({
            "model": "Isolation Forest (Unsupervised)",
            "threshold": 0.50,
            "roc_auc": round(float(roc_auc_score(y_test, iso_scores)), 4),
            "pr_auc": round(float(average_precision_score(y_test, iso_scores)), 4),
            "precision": round(float(precision_score(y_test, iso_flags, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, iso_flags, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, iso_flags, zero_division=0)), 4),
            "false_positives": int(fp_i),
            "false_negatives": int(fn_i),
            "true_positives": int(tp_i),
            "true_negatives": int(tn_i),
            "false_positive_rate": round(float(fp_i / (fp_i + tn_i)), 4),
            "false_negative_rate": round(float(fn_i / (fn_i + tp_i)), 4),
            "flagged_transactions": int(iso_flags.sum()),
            "estimated_business_cost_usd": int(fn_i * 1500 + fp_i * 35),
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(MODEL_COMPARISON_FILE, index=False)
    logger.info("Saved final model comparison table to %s", MODEL_COMPARISON_FILE)

    print("\n" + "=" * 95)
    print("FINAL MODEL COMPARISON BENCHMARK ON HELD-OUT TEST DATA")
    print("=" * 95)
    display_cols = ["model", "roc_auc", "pr_auc", "precision", "recall", "f1", "false_positives", "false_negatives", "estimated_business_cost_usd"]
    print(comp_df[display_cols].to_string(index=False))
    print("=" * 95 + "\n")

    return comp_df


if __name__ == "__main__":
    evaluate_test_set()
