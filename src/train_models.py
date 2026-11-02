"""
train_models.py
---------------
Model training pipeline with SMOTE imbalance handling, 5-fold cross-validation,
and model persistence (Logistic Regression, Random Forest, XGBoost).
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    CV_FOLDS,
    CV_SCORING,
    LOG_FILE,
    LOG_LEVEL,
    LOGISTIC_MODEL_FILE,
    LR_PARAMS,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    RF_MODEL_FILE,
    RF_PARAMS,
    TARGET_COLUMN,
    TRAIN_FILE,
    VALIDATION_FILE,
    XGB_MODEL_FILE,
    XGB_PARAMS,
)
from src.feature_engineering import engineer_features
from src.preprocessing import fit_pipeline, transform

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger("train_models")
np.random.seed(RANDOM_SEED)


def apply_smote(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE exclusively to the training split.
    Synthetic samples are synthesized strictly in the preprocessed feature space.
    """
    orig_counts = dict(zip(*np.unique(y, return_counts=True)))
    logger.info("Applying SMOTE to training set. Original class counts: %s", orig_counts)
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5, sampling_strategy=0.25)
    X_res, y_res = smote.fit_resample(X, y)
    res_counts = dict(zip(*np.unique(y_res, return_counts=True)))
    logger.info("Class counts after SMOTE (target ratio 0.25): %s", res_counts)
    return X_res, y_res


def run_cross_validation(model, X: np.ndarray, y: np.ndarray, model_name: str) -> Dict:
    """5-Fold Stratified Cross-Validation on the training set."""
    logger.info("Executing 5-Fold Stratified CV for %s ...", model_name)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results = cross_validate(
        model, X, y, cv=cv, scoring=CV_SCORING, return_train_score=False, n_jobs=-1
    )
    summary = {
        "model": model_name,
        "mean_roc_auc": round(float(np.mean(results["test_roc_auc"])), 4),
        "std_roc_auc":  round(float(np.std(results["test_roc_auc"])), 4),
        "mean_pr_auc":  round(float(np.mean(results["test_average_precision"])), 4),
        "std_pr_auc":   round(float(np.std(results["test_average_precision"])), 4),
        "mean_f1":      round(float(np.mean(results["test_f1"])), 4),
        "mean_recall":  round(float(np.mean(results["test_recall"])), 4),
        "mean_precision": round(float(np.mean(results["test_precision"])), 4),
    }
    logger.info("CV Summary [%s]: ROC-AUC=%.4f (+/- %.4f) | PR-AUC=%.4f | Recall=%.4f | F1=%.4f",
                model_name, summary["mean_roc_auc"], summary["std_roc_auc"],
                summary["mean_pr_auc"], summary["mean_recall"], summary["mean_f1"])
    return summary


def train_all_models() -> Dict:
    """
    Full training workflow:
    1. Load raw splits & run feature engineering
    2. Fit preprocessing pipeline
    3. Apply SMOTE to training partition
    4. Train Logistic Regression, Random Forest, and XGBoost
    5. Save artifacts to models/
    """
    logger.info("Loading training and validation sets...")
    train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
    val_df   = pd.read_csv(VALIDATION_FILE, low_memory=False)

    logger.info("Engineering features for training & validation partitions...")
    train_fe = engineer_features(train_df)
    val_fe   = engineer_features(val_df)

    logger.info("Fitting preprocessing pipeline on training data...")
    pipeline, X_train_raw, y_train_raw = fit_pipeline(train_fe)
    X_val, y_val = transform(pipeline, val_fe)

    logger.info("Applying SMOTE to training partition...")
    X_train_res, y_train_res = apply_smote(X_train_raw, y_train_raw)

    cv_summaries = []

    # 1. Logistic Regression
    logger.info("Training Logistic Regression...")
    lr_model = LogisticRegression(**LR_PARAMS)
    lr_cv = run_cross_validation(lr_model, X_train_raw, y_train_raw, "Logistic Regression")
    cv_summaries.append(lr_cv)
    lr_model.fit(X_train_res, y_train_res)
    joblib.dump(lr_model, LOGISTIC_MODEL_FILE)
    logger.info("Logistic Regression saved to %s", LOGISTIC_MODEL_FILE)

    # 2. Random Forest
    logger.info("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(**RF_PARAMS)
    rf_cv = run_cross_validation(rf_model, X_train_raw, y_train_raw, "Random Forest")
    cv_summaries.append(rf_cv)
    rf_model.fit(X_train_res, y_train_res)
    joblib.dump(rf_model, RF_MODEL_FILE)
    logger.info("Random Forest saved to %s", RF_MODEL_FILE)

    # 3. XGBoost Classifier (Primary High-Performance Model)
    logger.info("Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_cv = run_cross_validation(xgb_model, X_train_raw, y_train_raw, "XGBoost")
    cv_summaries.append(xgb_cv)
    xgb_model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    joblib.dump(xgb_model, XGB_MODEL_FILE)
    logger.info("XGBoost saved to %s", XGB_MODEL_FILE)

    # Save Cross-Validation Summary Report
    cv_df = pd.DataFrame(cv_summaries)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_report_path = REPORTS_DIR / "cross_validation_results.csv"
    cv_df.to_csv(cv_report_path, index=False)
    logger.info("Cross-validation report saved to %s", cv_report_path)

    return {
        "lr": lr_model,
        "rf": rf_model,
        "xgb": xgb_model,
        "cv_results": cv_df,
    }


if __name__ == "__main__":
    train_all_models()
