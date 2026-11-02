"""
anomaly_detection.py
--------------------
Unsupervised anomaly detection module using Isolation Forest.
Identifies suspicious transactions without labels, providing an independent
behavioral risk signal and detecting zero-day fraud mechanisms.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    ISO_MODEL_FILE,
    ISO_PARAMS,
    LOG_FILE,
    LOG_LEVEL,
    MODELS_DIR,
    RANDOM_SEED,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger("anomaly_detection")


def train_isolation_forest(X_train: np.ndarray) -> IsolationForest:
    """
    Train Isolation Forest model purely in an unsupervised fashion (no fraud labels).
    Persists model to models/isolation_forest.pkl.
    """
    logger.info("Training Isolation Forest on %d samples (features=%d)...", len(X_train), X_train.shape[1])
    iso = IsolationForest(**ISO_PARAMS)
    iso.fit(X_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso, ISO_MODEL_FILE)
    logger.info("Isolation Forest successfully trained and saved to %s", ISO_MODEL_FILE)
    return iso


def load_isolation_forest() -> IsolationForest:
    """Load persisted Isolation Forest model."""
    if not ISO_MODEL_FILE.exists():
        raise FileNotFoundError(f"Isolation Forest model not found at {ISO_MODEL_FILE}")
    return joblib.load(ISO_MODEL_FILE)


def get_anomaly_scores(iso: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Generate normalized anomaly scores in [0.0, 1.0].
    IsolationForest decision_function outputs negative values for anomalies.
    We invert and min-max scale so higher value = more anomalous.
    """
    raw_scores = iso.decision_function(X) # lower = more anomalous
    inverted = -raw_scores

    # Min-max normalization
    min_val, max_val = inverted.min(), inverted.max()
    if max_val > min_val:
        normalized = (inverted - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(inverted)

    return np.round(normalized, 4)


def get_anomaly_flags(iso: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Generate binary anomaly flags (1 = anomaly, 0 = normal).
    predict() returns -1 for anomaly, +1 for inlier.
    """
    preds = iso.predict(X)
    return np.where(preds == -1, 1, 0)


def evaluate_iso_vs_labels(iso_flags: np.ndarray, iso_scores: np.ndarray, y_true: np.ndarray) -> Dict:
    """
    Benchmark unsupervised anomaly signals against ground-truth labels.
    """
    auc = roc_auc_score(y_true, iso_scores)
    logger.info("Unsupervised Isolation Forest ROC-AUC vs Ground Truth: %.4f", auc)
    report = classification_report(y_true, iso_flags, target_names=["Normal", "Anomaly"], output_dict=True)
    logger.info("Anomaly Detection Precision: %.4f | Recall: %.4f",
                report["Anomaly"]["precision"], report["Anomaly"]["recall"])
    return {
        "roc_auc": round(auc, 4),
        "precision": round(report["Anomaly"]["precision"], 4),
        "recall": round(report["Anomaly"]["recall"], 4),
        "f1": round(report["Anomaly"]["f1-score"], 4),
    }


if __name__ == "__main__":
    from src.config import TEST_FILE, TRAIN_FILE
    from src.feature_engineering import engineer_features
    from src.preprocessing import load_pipeline, transform

    print("Running Unsupervised Anomaly Detection pipeline independently...")
    train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
    test_df  = pd.read_csv(TEST_FILE, low_memory=False)

    train_fe = engineer_features(train_df)
    test_fe  = engineer_features(test_df)

    pipeline = load_pipeline()
    X_train, _ = transform(pipeline, train_fe)
    X_test, y_test = transform(pipeline, test_fe)

    iso = train_isolation_forest(X_train)
    iso_scores = get_anomaly_scores(iso, X_test)
    iso_flags  = get_anomaly_flags(iso, X_test)

    eval_results = evaluate_iso_vs_labels(iso_flags, iso_scores, y_test)
    print("\n[SUCCESS] Isolation Forest Evaluation vs Ground Truth:")
    for k, v in eval_results.items():
        print(f"  {k:15s}: {v}")
    print(f"Anomaly Detection Flag Rate: {iso_flags.mean() * 100:.2f}%")
