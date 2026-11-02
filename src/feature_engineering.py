"""
feature_engineering.py
----------------------
Feature engineering module for the Intelligent Banking Transaction & Fraud Detection System.
Generates transaction, velocity, behavioral, temporal, and security risk signals
without target leakage.
"""

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    ALL_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    LOG_FILE,
    LOG_LEVEL,
    TARGET_COLUMN,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger("feature_engineering")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all engineered features for train, val, test, or live transactions.
    Ensures zero target leakage.
    """
    df = df.copy()

    # 1. Transaction Amount Signals
    df["log_transaction_amount"] = np.log1p(np.maximum(df["transaction_amount"], 0))
    df["amount_to_balance_ratio"] = df["transaction_amount"] / (df["account_balance"] + 1e-5)
    df["amount_deviation_from_customer_average"] = (
        (df["transaction_amount"] - df["average_transaction_amount"])
        / (df["average_transaction_amount"] + 1e-5)
    )
    df["amount_deviation_from_previous_transaction"] = (
        (df["transaction_amount"] - df["previous_transaction_amount"])
        / (df["previous_transaction_amount"] + 1e-5)
    )

    # 2. Cyclic Temporal Encodings & Velocity Projections
    hour = df["transaction_hour"]
    day = df["transaction_day"]
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["day_sin"]  = np.sin(2 * np.pi * day / 31.0)
    df["day_cos"]  = np.cos(2 * np.pi * day / 31.0)

    # Time since previous transaction (seconds proxy)
    df["time_since_previous_transaction"] = np.maximum(
        15.0, 86400.0 / (np.maximum(df["transaction_velocity"], 0.1))
    )

    # Multi-window velocity features
    df["transactions_last_10_minutes"] = np.round(np.maximum(0, df["transaction_velocity"] * 0.15))
    df["transactions_last_1_hour"]     = np.round(np.maximum(0, df["transaction_velocity"] * 0.35))
    df["transactions_last_24_hours"]   = df["transaction_velocity"]
    df["amount_last_1_hour"]           = np.round(df["transactions_last_1_hour"] * df["transaction_amount"] * 0.65, 2)
    df["amount_last_24_hours"]         = np.round(df["transactions_last_24_hours"] * df["average_transaction_amount"] * 1.1, 2)

    # 3. Behavioral and Security Signal Ratios
    df["failed_attempt_ratio"] = df["failed_transaction_count"] / (df["login_attempts"] + 1e-5)
    daily_freq_norm = np.maximum(df["transaction_frequency"] / 30.0, 0.1)
    df["velocity_surge_ratio"] = df["transaction_velocity"] / daily_freq_norm
    df["balance_drain_ratio"]  = np.clip(df["transaction_amount"] / (df["account_balance"] + 1e-5), 0, 5.0)

    # Unusual auth flag: SMS OTP combined with new device (high risk SIM swap/intercept signal)
    df["unusual_auth_flag"] = (
        (df["authentication_method"] == "SMS OTP") & (df["new_device"] == 1)
    ).astype(int)

    # 4. Behavioral Risk Heuristic Proxy (0.0 to 1.0) - Unsupervised risk prior
    amount_dev_norm = np.clip(df["amount_deviation_from_customer_average"] / 3.0, 0.0, 1.0)
    failed_norm     = np.clip(df["failed_attempt_ratio"], 0.0, 1.0)
    surge_norm      = np.clip(df["velocity_surge_ratio"] / 3.0, 0.0, 1.0)

    df["behavioral_risk_proxy"] = (
        0.25 * amount_dev_norm +
        0.20 * df["new_device"].astype(float) +
        0.15 * df["new_location"].astype(float) +
        0.15 * failed_norm +
        0.15 * surge_norm +
        0.10 * df["night_transaction"].astype(float)
    )

    return df


def get_feature_columns() -> List[str]:
    """Return all feature names expected by the model."""
    return ALL_NUMERIC_FEATURES + CATEGORICAL_FEATURES


if __name__ == "__main__":
    from src.config import SAMPLE_FILE
    print(f"Loading sample data from {SAMPLE_FILE} ...")
    sample_df = pd.read_csv(SAMPLE_FILE)
    fe_df = engineer_features(sample_df)
    print(f"[SUCCESS] Feature engineering complete! Original columns: {sample_df.shape[1]} -> New columns: {fe_df.shape[1]}")
    engineered_cols = [c for c in fe_df.columns if c not in sample_df.columns]
    print("\nEngineered Feature Sample:")
    print(fe_df[["transaction_id", "transaction_amount", "amount_to_balance_ratio", "velocity_surge_ratio", "behavioral_risk_proxy"]].head())
