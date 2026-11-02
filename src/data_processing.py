"""
data_processing.py
------------------
Data quality, validation, cleaning, and chronological train/validation/test splitting
for the Intelligent Banking Transaction & Fraud Detection System.
"""

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    CLEAN_DATA_FILE,
    LOG_FILE,
    LOG_LEVEL,
    PROCESSED_DIR,
    RAW_DATA_FILE,
    TARGET_COLUMN,
    TEST_FILE,
    TEST_RATIO,
    TRAIN_FILE,
    TRAIN_RATIO,
    VALIDATION_FILE,
    VALIDATION_RATIO,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger("data_processing")


def load_raw_data(filepath: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load raw banking transaction CSV with type enforcement."""
    logger.info("Loading raw dataset from %s ...", filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")

    df = pd.read_csv(filepath, low_memory=False)
    logger.info("Raw data loaded: %d rows, %d columns.", len(df), df.shape[1])
    return df


def audit_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform rigorous data quality audits and cleaning:
    - Null value checking & imputation
    - Duplicate detection & removal
    - Amount validation (amount > 0)
    - Timestamp parsing & sorting
    - Inconsistent categorical standardization
    """
    logger.info("Starting data quality audits...")
    initial_rows = len(df)

    # 1. Check & handle missing values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    logger.info("Total missing values across all columns: %d", total_nulls)
    if total_nulls > 0:
        # Impute numeric with median, categoricals with mode
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in [np.float64, np.int64]:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

    # 2. Check & drop duplicate transaction IDs
    duplicate_count = df.duplicated(subset=["transaction_id"]).sum()
    logger.info("Duplicate transactions detected: %d", duplicate_count)
    if duplicate_count > 0:
        df = df.drop_duplicates(subset=["transaction_id"], keep="first")

    # 3. Validate transaction amounts (must be strictly positive)
    invalid_amounts = (df["transaction_amount"] <= 0).sum()
    if invalid_amounts > 0:
        logger.warning("Found %d non-positive transaction amounts. Filtering out...", invalid_amounts)
        df = df[df["transaction_amount"] > 0]

    # 4. Standardize timestamps and sort chronologically
    df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"])
    df = df.sort_values(by="transaction_timestamp").reset_index(drop=True)

    # 5. Clean string whitespace
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    logger.info("Data cleaning complete. Rows retained: %d (%.2f%%)",
                len(df), (len(df) / initial_rows) * 100)
    return df


def split_chronological(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VALIDATION_RATIO,
    test_ratio: float = TEST_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-aware chronological split to avoid temporal data leakage in transaction data.
    70% Train -> 15% Validation -> 15% Test
    """
    logger.info("Executing chronological train/validation/test split...")
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df   = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df  = df.iloc[val_end:].copy().reset_index(drop=True)

    logger.info("Train set: %d rows (Fraud rate: %.3f%% | %s to %s)",
                len(train_df), train_df[TARGET_COLUMN].mean() * 100,
                train_df['transaction_timestamp'].min(), train_df['transaction_timestamp'].max())
    logger.info("Val set:   %d rows (Fraud rate: %.3f%% | %s to %s)",
                len(val_df), val_df[TARGET_COLUMN].mean() * 100,
                val_df['transaction_timestamp'].min(), val_df['transaction_timestamp'].max())
    logger.info("Test set:  %d rows (Fraud rate: %.3f%% | %s to %s)",
                len(test_df), test_df[TARGET_COLUMN].mean() * 100,
                test_df['transaction_timestamp'].min(), test_df['transaction_timestamp'].max())

    return train_df, val_df, test_df


def run_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete data extraction, cleaning, and persistence pipeline."""
    df_raw = load_raw_data(RAW_DATA_FILE)
    df_clean = audit_and_clean_data(df_raw)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(CLEAN_DATA_FILE, index=False)
    logger.info("Clean dataset saved to %s", CLEAN_DATA_FILE)

    train_df, val_df, test_df = split_chronological(df_clean)

    train_df.to_csv(TRAIN_FILE, index=False)
    val_df.to_csv(VALIDATION_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    logger.info("Split datasets saved: train.csv, validation.csv, test.csv")
    return train_df, val_df, test_df


if __name__ == "__main__":
    run_pipeline()
