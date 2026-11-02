"""
preprocessing.py
----------------
Scikit-learn preprocessing pipeline for the Intelligent Banking Transaction & Fraud Detection System.
Handles scaling of numeric signals and encoding of categorical variables.
Persists the fitted pipeline for deterministic inference.
"""

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    ALL_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_LIST_FILE,
    LOG_FILE,
    LOG_LEVEL,
    MODELS_DIR,
    PIPELINE_FILE,
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
logger = logging.getLogger("preprocessing")


def build_pipeline(
    numeric_cols: List[str] = ALL_NUMERIC_FEATURES,
    categorical_cols: List[str] = CATEGORICAL_FEATURES,
) -> Pipeline:
    """Build unfitted sklearn ColumnTransformer pipeline."""
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    return pipeline


def fit_pipeline(
    train_df: pd.DataFrame,
    numeric_cols: List[str] = ALL_NUMERIC_FEATURES,
    categorical_cols: List[str] = CATEGORICAL_FEATURES,
) -> Tuple[Pipeline, np.ndarray, np.ndarray]:
    """
    Fit the pipeline on training data and return (pipeline, X_train, y_train).
    """
    logger.info("Fitting preprocessing pipeline on %d training rows...", len(train_df))
    pipeline = build_pipeline(numeric_cols, categorical_cols)

    feature_cols = numeric_cols + categorical_cols
    X_train = pipeline.fit_transform(train_df[feature_cols])
    y_train = train_df[TARGET_COLUMN].values.astype(int) if TARGET_COLUMN in train_df else None

    # Save fitted pipeline and feature list
    save_pipeline(pipeline)
    save_feature_names(feature_cols)

    logger.info("Pipeline fitted successfully. Feature dimension: %d", X_train.shape[1])
    return pipeline, X_train, y_train


def transform(pipeline: Pipeline, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a dataset using the fitted pipeline."""
    feature_cols = ALL_NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = pipeline.transform(df[feature_cols])
    y = df[TARGET_COLUMN].values.astype(int) if TARGET_COLUMN in df else None
    return X, y


def save_pipeline(pipeline: Pipeline, filepath: Path = PIPELINE_FILE) -> None:
    """Persist pipeline object to disk."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, filepath)
    logger.info("Saved fitted pipeline to %s", filepath)


def load_pipeline(filepath: Path = PIPELINE_FILE) -> Pipeline:
    """Load persisted pipeline object."""
    if not filepath.exists():
        raise FileNotFoundError(f"Pipeline artifact not found at {filepath}")
    return joblib.load(filepath)


def save_feature_names(features: List[str], filepath: Path = FEATURE_LIST_FILE) -> None:
    """Write feature names list to text file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for feat in features:
            f.write(f"{feat}\n")
    logger.info("Saved feature list to %s (%d features)", filepath, len(features))


def get_feature_names(filepath: Path = FEATURE_LIST_FILE) -> List[str]:
    """Read saved feature names."""
    if not filepath.exists():
        return ALL_NUMERIC_FEATURES + CATEGORICAL_FEATURES
    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    from src.config import TRAIN_FILE
    from src.feature_engineering import engineer_features
    print(f"Loading training data from {TRAIN_FILE} ...")
    train_df = pd.read_csv(TRAIN_FILE, nrows=5000)
    train_fe = engineer_features(train_df)
    pipe, X_t, y_t = fit_pipeline(train_fe)
    print(f"[SUCCESS] Preprocessing pipeline fitted successfully!")
    print(f"Transformed matrix shape: {X_t.shape}")
    print(f"Saved pipeline to: {PIPELINE_FILE}")
