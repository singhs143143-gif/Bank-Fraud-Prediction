from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "best_fraud_model.pkl"
ISOLATION_MODEL_PATH = PROJECT_DIR / "isolation_forest_anomaly_model.pkl"
REFERENCE_DATA_PATH = PROJECT_DIR / "data_sample.csv"

REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def _engineer_features(transactions: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(transactions.columns)
    if missing:
        raise ValueError(f"Transactions are missing required columns: {sorted(missing)}")

    features = transactions.copy()
    if "nameOrig" not in features:
        features["nameOrig"] = ""
    if "nameDest" not in features:
        features["nameDest"] = ""

    features["hour"] = features["step"] % 24
    features["day_index"] = features["step"] // 24
    features["log_amount"] = np.log1p(features["amount"].clip(lower=0))
    features["is_high_amount"] = (
        features["amount"] >= features["amount"].quantile(0.99)
    ).astype("int8")
    features["origin_balance_change"] = (
        features["oldbalanceOrg"] - features["newbalanceOrig"]
    )
    features["destination_balance_change"] = (
        features["newbalanceDest"] - features["oldbalanceDest"]
    )
    features["origin_balance_error"] = (
        features["oldbalanceOrg"] - features["amount"] - features["newbalanceOrig"]
    )
    features["destination_balance_error"] = (
        features["oldbalanceDest"] + features["amount"] - features["newbalanceDest"]
    )
    features["origin_zero_after_txn"] = (features["newbalanceOrig"] == 0).astype("int8")
    features["destination_zero_before_txn"] = (
        features["oldbalanceDest"] == 0
    ).astype("int8")
    features["origin_txn_count_in_sample"] = (
        features["nameOrig"].map(features["nameOrig"].value_counts()).fillna(1).astype("int32")
    )
    features["destination_txn_count_in_sample"] = (
        features["nameDest"].map(features["nameDest"].value_counts()).fillna(1).astype("int32")
    )
    features["destination_is_merchant"] = (
        features["nameDest"].astype(str).str.startswith("M").astype("int8")
    )

    columns = REQUIRED_COLUMNS + [
        "hour",
        "day_index",
        "log_amount",
        "is_high_amount",
        "origin_balance_change",
        "destination_balance_change",
        "origin_balance_error",
        "destination_balance_error",
        "origin_zero_after_txn",
        "destination_zero_before_txn",
        "origin_txn_count_in_sample",
        "destination_txn_count_in_sample",
        "destination_is_merchant",
    ]
    return features[columns]


@lru_cache(maxsize=1)
def _amount_reference() -> np.ndarray:
    if not REFERENCE_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Amount reference data not found: {REFERENCE_DATA_PATH}"
        )
    amounts = pd.read_csv(REFERENCE_DATA_PATH, usecols=["amount"])["amount"]
    return np.sort(amounts.dropna().to_numpy(dtype=float))


def engineer_anomaly_features(transactions: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(transactions.columns)
    if missing:
        raise ValueError(f"Transactions are missing required columns: {sorted(missing)}")

    features = transactions.copy()
    if "nameOrig" not in features:
        features["nameOrig"] = [f"missing-origin-{i}" for i in range(len(features))]
    if "nameDest" not in features:
        features["nameDest"] = [f"missing-destination-{i}" for i in range(len(features))]

    amount = features["amount"].clip(lower=0)
    origin_balance = features["oldbalanceOrg"].clip(lower=0)
    destination_balance = features["oldbalanceDest"].clip(lower=0)
    hour = features["step"] % 24
    day_index = features["step"] // 24
    origin_change = features["oldbalanceOrg"] - features["newbalanceOrig"]
    destination_change = features["newbalanceDest"] - features["oldbalanceDest"]
    origin_error = origin_balance - amount - features["newbalanceOrig"]
    destination_error = destination_balance + amount - features["newbalanceDest"]
    amount_reference = _amount_reference()

    features["log_amount"] = np.log1p(amount)
    features["amount_percentile"] = (
        np.searchsorted(amount_reference, amount.to_numpy(), side="right")
        / max(len(amount_reference), 1)
    )
    features["is_high_amount"] = (
        amount >= np.quantile(amount_reference, 0.99)
    ).astype("int8")
    features["amount_to_orig_ratio"] = amount / (origin_balance + 1e-9)
    features["amount_to_dest_ratio"] = amount / (destination_balance + 1e-9)
    features["log_oldbalanceOrg"] = np.log1p(origin_balance)
    features["log_newbalanceOrig"] = np.log1p(features["newbalanceOrig"].clip(lower=0))
    features["log_oldbalanceDest"] = np.log1p(destination_balance)
    features["log_newbalanceDest"] = np.log1p(features["newbalanceDest"].clip(lower=0))
    features["hour"] = hour
    features["day_index"] = day_index
    features["is_night"] = ((hour < 6) | (hour >= 22)).astype("int8")
    features["is_weekend"] = ((day_index % 7) >= 5).astype("int8")
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["origin_balance_change"] = origin_change
    features["destination_balance_change"] = destination_change
    features["origin_balance_error"] = origin_error
    features["destination_balance_error"] = destination_error
    features["origin_balance_error_abs"] = origin_error.abs()
    features["destination_balance_error_abs"] = destination_error.abs()
    features["origin_zero_after_txn"] = (features["newbalanceOrig"] == 0).astype("int8")
    features["destination_zero_before_txn"] = (
        features["oldbalanceDest"] == 0
    ).astype("int8")
    features["full_drain_flag"] = (
        (features["newbalanceOrig"] == 0) & (origin_balance > 0)
    ).astype("int8")
    features["orig_balance_drain_ratio"] = amount / (origin_balance + 1e-9)
    features["origin_is_customer"] = (
        features["nameOrig"].astype(str).str.startswith("C").astype("int8")
    )
    features["destination_is_merchant"] = (
        features["nameDest"].astype(str).str.startswith("M").astype("int8")
    )
    features["destination_is_customer"] = (
        features["nameDest"].astype(str).str.startswith("C").astype("int8")
    )

    origin_groups = features.groupby("nameOrig", sort=False)
    features["orig_txn_count"] = origin_groups["amount"].transform("count")
    features["orig_avg_amount"] = origin_groups["amount"].transform("mean")
    features["orig_amount_deviation"] = amount - features["orig_avg_amount"]
    features["orig_amount_to_mean_ratio"] = amount / (features["orig_avg_amount"] + 1e-9)

    destination_groups = features.groupby("nameDest", sort=False)
    features["dest_txn_count"] = destination_groups["amount"].transform("count")
    features["dest_avg_amount"] = destination_groups["amount"].transform("mean")

    recent_counts = np.ones(len(features), dtype="int32")
    for indexes in origin_groups.groups.values():
        positions = features.index.get_indexer(indexes)
        steps = features.iloc[positions]["step"].to_numpy()
        order = np.argsort(steps)
        ordered_steps = steps[order]
        left_edges = np.searchsorted(ordered_steps, ordered_steps - 24, side="left")
        recent_counts[positions[order]] = np.arange(len(order)) + 1 - left_edges
    features["orig_txns_last_24_steps"] = recent_counts

    transaction_types = features["type"].astype(str)
    for transaction_type in ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"):
        features[f"type_{transaction_type}"] = (
            transaction_types == transaction_type
        ).astype("int8")
    features["is_high_risk_type"] = transaction_types.isin(
        ["TRANSFER", "CASH_OUT"]
    ).astype("int8")

    columns = [
        "amount",
        "log_amount",
        "amount_percentile",
        "is_high_amount",
        "amount_to_orig_ratio",
        "amount_to_dest_ratio",
        "log_oldbalanceOrg",
        "log_newbalanceOrig",
        "log_oldbalanceDest",
        "log_newbalanceDest",
        "step",
        "hour",
        "day_index",
        "is_night",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "origin_balance_change",
        "destination_balance_change",
        "origin_balance_error",
        "destination_balance_error",
        "origin_balance_error_abs",
        "destination_balance_error_abs",
        "origin_zero_after_txn",
        "destination_zero_before_txn",
        "full_drain_flag",
        "orig_balance_drain_ratio",
        "origin_is_customer",
        "orig_txn_count",
        "orig_avg_amount",
        "orig_amount_deviation",
        "orig_amount_to_mean_ratio",
        "destination_is_merchant",
        "destination_is_customer",
        "dest_txn_count",
        "dest_avg_amount",
        "orig_txns_last_24_steps",
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER",
        "is_high_risk_type",
    ]
    return features[columns]


class _FeatureEngineer:
    def transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        return _engineer_features(transactions)


@lru_cache(maxsize=1)
def load_artifacts() -> dict:
    for artifact_path in (MODEL_PATH, ISOLATION_MODEL_PATH):
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Required model artifact not found: {artifact_path}")

    model_artifact = joblib.load(MODEL_PATH)
    isolation_artifact = joblib.load(ISOLATION_MODEL_PATH)
    isolation_model = isolation_artifact["model"]

    return {
        "model": model_artifact["model"],
        "threshold": float(model_artifact.get("threshold", 0.5)),
        "feature_engineer": _FeatureEngineer(),
        "iso_model": isolation_model,
        "iso_imputer": isolation_artifact["imputer"],
        "iso_scaler": isolation_artifact["scaler"],
        "iso_numeric_features": isolation_artifact["numeric_features"],
        "iso_min": float(isolation_artifact["score_min"]),
        "iso_max": float(isolation_artifact["score_max"]),
    }


def predict_transaction(transaction: dict) -> dict:
    if 'transaction_amount' in transaction or 'transaction_id' in transaction:
        from src.prediction import predict_transaction as src_predict
        return src_predict(transaction)
    artifacts = load_artifacts()
    raw = pd.DataFrame([transaction])
    features = artifacts["feature_engineer"].transform(raw)

    fraud_probability = float(artifacts["model"].predict_proba(features)[0, 1])
    anomaly_features = engineer_anomaly_features(raw)
    numeric_features = anomaly_features[artifacts["iso_numeric_features"]]
    scaled_features = artifacts["iso_scaler"].transform(
        artifacts["iso_imputer"].transform(numeric_features)
    )
    raw_anomaly_score = -float(artifacts["iso_model"].score_samples(scaled_features)[0])
    anomaly_score = float(
        np.clip(
            (raw_anomaly_score - artifacts["iso_min"])
            / (artifacts["iso_max"] - artifacts["iso_min"]),
            0.0,
            1.0,
        )
    )

    reasons = []
    if transaction["type"] in {"TRANSFER", "CASH_OUT"}:
        reasons.append(f"{transaction['type']} transactions are higher risk in this dataset")
    if transaction["newbalanceOrig"] == 0 and transaction["amount"] > 0:
        reasons.append("The origin account is empty after the transaction")
    if abs(
        transaction["oldbalanceOrg"]
        - transaction["amount"]
        - transaction["newbalanceOrig"]
    ) > 1:
        reasons.append("The origin balance change does not match the transaction amount")
    if not reasons:
        reasons.append("No additional balance-pattern warning was detected")

    return {
        "fraud_probability": fraud_probability,
        "anomaly_score": anomaly_score,
        "decision_threshold": artifacts["threshold"],
        "risk_segment": (
            "Critical"
            if fraud_probability >= 0.80 or (fraud_probability >= 0.30 and anomaly_score >= 0.65)
            else "High"
            if fraud_probability >= 0.15 or (fraud_probability >= 0.02 and anomaly_score >= 0.60)
            else "Medium"
            if fraud_probability >= 0.01 or anomaly_score >= 0.50
            else "Low"
        ),
        "top_reasons": reasons,
    }
