"""
config.py
---------
Central configuration for the Intelligent Banking Transaction & Fraud Detection System.
All paths, schemas, feature definitions, hyperparameters, and thresholds live here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project Directories
# ---------------------------------------------------------------------------
ROOT_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
SAMPLE_DIR      = DATA_DIR / "sample"
MODELS_DIR      = ROOT_DIR / "models"
REPORTS_DIR     = ROOT_DIR / "reports"
LOGS_DIR        = ROOT_DIR / "logs"
SQL_DIR         = ROOT_DIR / "sql"
DASHBOARD_DIR   = ROOT_DIR / "dashboard"
NOTEBOOKS_DIR   = ROOT_DIR / "notebooks"

# Ensure runtime directories exist
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, SAMPLE_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR, SQL_DIR, DASHBOARD_DIR, NOTEBOOKS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------------------
RAW_DATA_FILE   = RAW_DIR / "banking_transactions_raw.csv"
SAMPLE_FILE     = SAMPLE_DIR / "data_sample.csv"

CLEAN_DATA_FILE = PROCESSED_DIR / "banking_transactions_clean.csv"
TRAIN_FILE      = PROCESSED_DIR / "train.csv"
VALIDATION_FILE = PROCESSED_DIR / "validation.csv"
TEST_FILE       = PROCESSED_DIR / "test.csv"
FEATURES_FILE   = PROCESSED_DIR / "features_engineered.csv"

# Model Artifacts
LOGISTIC_MODEL_FILE   = MODELS_DIR / "logistic_regression.pkl"
RF_MODEL_FILE         = MODELS_DIR / "random_forest.pkl"
XGB_MODEL_FILE        = MODELS_DIR / "xgboost.pkl"
ISO_MODEL_FILE        = MODELS_DIR / "isolation_forest.pkl"
PIPELINE_FILE         = MODELS_DIR / "preprocessing_pipeline.pkl"
FEATURE_LIST_FILE     = MODELS_DIR / "feature_list.txt"
THRESHOLD_CONFIG_FILE = MODELS_DIR / "threshold_config.json"

# Reports
MODEL_COMPARISON_FILE   = REPORTS_DIR / "model_comparison.csv"
FRAUD_PREDICTIONS_FILE  = REPORTS_DIR / "fraud_predictions.csv"
FEATURE_IMPORTANCE_FILE = REPORTS_DIR / "feature_importance.csv"
ALERTS_FILE             = REPORTS_DIR / "fraud_alerts.csv"

# Logging
LOG_FILE  = LOGS_DIR / "fraud_system.log"
LOG_LEVEL = "INFO"

# ---------------------------------------------------------------------------
# Pipeline Constants & Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED       = 42
TOTAL_RECORDS     = 300_000
TARGET_COLUMN     = "fraud_label"

# Chronological Time-Aware Splits
TRAIN_RATIO       = 0.70
VALIDATION_RATIO  = 0.15
TEST_RATIO        = 0.15

# ---------------------------------------------------------------------------
# Feature Column Specifications
# ---------------------------------------------------------------------------
METADATA_COLUMNS = [
    "transaction_id",
    "customer_id",
    "merchant_id",
    "device_id",
    "ip_address",
    "transaction_timestamp",
    "transaction_location",
    "customer_location",
]

CATEGORICAL_FEATURES = [
    "transaction_type",
    "merchant_category",
    "device_type",
    "payment_method",
    "authentication_method",
]

NUMERIC_BASE_FEATURES = [
    "transaction_amount",
    "account_age_days",
    "customer_age",
    "customer_income",
    "account_balance",
    "previous_transaction_amount",
    "transaction_frequency",
    "failed_transaction_count",
    "login_attempts",
    "international_transaction",
    "weekend_transaction",
    "night_transaction",
    "new_device",
    "new_location",
    "distance_from_home",
    "transaction_velocity",
    "customer_transaction_count",
    "average_transaction_amount",
    "transaction_hour",
    "transaction_day",
    "transaction_month",
]

ENGINEERED_NUMERIC_FEATURES = [
    "log_transaction_amount",
    "amount_to_balance_ratio",
    "amount_deviation_from_customer_average",
    "amount_deviation_from_previous_transaction",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "time_since_previous_transaction",
    "transactions_last_10_minutes",
    "transactions_last_1_hour",
    "transactions_last_24_hours",
    "amount_last_1_hour",
    "amount_last_24_hours",
    "failed_attempt_ratio",
    "velocity_surge_ratio",
    "unusual_auth_flag",
    "balance_drain_ratio",
    "behavioral_risk_proxy",
]

ALL_NUMERIC_FEATURES = NUMERIC_BASE_FEATURES + ENGINEERED_NUMERIC_FEATURES

# ---------------------------------------------------------------------------
# Risk Tiers & Alert Configuration
# ---------------------------------------------------------------------------
RISK_THRESHOLDS = {
    "low_max"    : 0.30,   # 0.00 – 0.299 → LOW RISK
    "medium_max" : 0.70,   # 0.30 – 0.699 → MEDIUM RISK
    # >= 0.70             → HIGH RISK
}

ALERT_THRESHOLD = 70      # Trigger operational fraud alert if risk_score >= 70
OPTIMAL_CLASSIFICATION_THRESHOLD = 0.40

# ---------------------------------------------------------------------------
# Machine Learning Hyperparameters
# ---------------------------------------------------------------------------
LR_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}

RF_PARAMS = {
    "n_estimators": 150,
    "max_depth": 14,
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

XGB_PARAMS = {
    "n_estimators": 250,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 4,
    "gamma": 0.5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 25,    # Imbalance correction (roughly (1 - p) / p)
    "eval_metric": "aucpr",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

ISO_PARAMS = {
    "n_estimators": 150,
    "max_samples": "auto",
    "contamination": 0.025,    # Approximate 2.5% anomaly baseline
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Cross-Validation
CV_FOLDS = 5
CV_SCORING = ["roc_auc", "average_precision", "f1", "precision", "recall"]
