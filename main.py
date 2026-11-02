"""
main.py
-------
Central orchestration CLI for the Intelligent Banking Transaction & Fraud Detection System.

Usage:
------
python main.py                  # Run entire end-to-end pipeline
python main.py --stage data     # Data validation, cleaning, and chronological splitting
python main.py --stage train    # Feature engineering, SMOTE, and supervised model training
python main.py --stage anomaly  # Unsupervised Isolation Forest training
python main.py --stage evaluate # Test set evaluation, metrics, threshold optimization
python main.py --stage predict  # Live inference demo and batch scoring
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Fix Windows console UTF-8 encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    ALERTS_FILE,
    FRAUD_PREDICTIONS_FILE,
    LOG_FILE,
    LOG_LEVEL,
    MODEL_COMPARISON_FILE,
    RAW_DATA_FILE,
    REPORTS_DIR,
    TEST_FILE,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


def stage_data():
    logger.info("-" * 60)
    logger.info("STAGE 1 - DATA QUALITY, CLEANING & CHRONOLOGICAL SPLIT")
    logger.info("-" * 60)
    from src.data_processing import run_pipeline
    train, val, test = run_pipeline()
    logger.info("[SUCCESS] Data stage complete: Train=%d, Val=%d, Test=%d", len(train), len(val), len(test))
    return train, val, test


def stage_train():
    logger.info("-" * 60)
    logger.info("STAGE 2 - FEATURE ENGINEERING, SMOTE & MODEL TRAINING")
    logger.info("-" * 60)
    from src.train_models import train_all_models
    artifacts = train_all_models()
    logger.info("[SUCCESS] Supervised model training and 5-fold CV complete.")
    return artifacts


def stage_anomaly():
    logger.info("-" * 60)
    logger.info("STAGE 3 - UNSUPERVISED ISOLATION FOREST ANOMALY DETECTION")
    logger.info("-" * 60)
    import pandas as pd
    from src.anomaly_detection import (
        evaluate_iso_vs_labels,
        get_anomaly_flags,
        get_anomaly_scores,
        train_isolation_forest,
    )
    from src.config import TARGET_COLUMN, TEST_FILE, TRAIN_FILE
    from src.feature_engineering import engineer_features
    from src.preprocessing import load_pipeline, transform

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
    logger.info("[SUCCESS] Isolation Forest anomaly detection complete: Anomaly Rate=%.2f%%", iso_flags.mean() * 100)
    return iso, iso_scores, iso_flags


def stage_evaluate():
    logger.info("-" * 60)
    logger.info("STAGE 4 - COMPREHENSIVE TEST SET BENCHMARK & THRESHOLD TUNING")
    logger.info("-" * 60)
    from src.evaluation import evaluate_test_set
    comp_df = evaluate_test_set()
    logger.info("[SUCCESS] Model evaluation and threshold tuning complete.")
    return comp_df


def stage_predict():
    logger.info("-" * 60)
    logger.info("STAGE 5 - REAL-TIME INFERENCE DEMO & BATCH SCORING")
    logger.info("-" * 60)
    import pandas as pd
    from src.alert_system import create_alert, generate_batch_alerts, save_alerts
    from src.prediction import predict_batch, predict_transaction

    # 1. Single Transaction Real-Time Scoring (High Risk Scenario)
    high_risk_txn = {
        "transaction_id": "TXN_LIVE_99841",
        "customer_id": "CUST_001420",
        "transaction_timestamp": "2024-06-25 02:45:10",
        "transaction_amount": 9450.00,
        "transaction_type": "TRANSFER",
        "merchant_category": "Luxury Goods",
        "merchant_id": "MERCH_00088",
        "account_age_days": 1200,
        "customer_age": 42,
        "customer_income": 65000.0,
        "account_balance": 10200.0,
        "transaction_location": "Singapore, SGP",
        "customer_location": "New York, USA",
        "device_id": "DEV_NEW_7721",
        "device_type": "Mobile Android",
        "payment_method": "Wire Transfer",
        "authentication_method": "SMS OTP",
        "ip_address": "194.67.210.4",
        "previous_transaction_amount": 85.00,
        "transaction_frequency": 22,
        "failed_transaction_count": 2,
        "login_attempts": 3,
        "international_transaction": 1,
        "weekend_transaction": 0,
        "night_transaction": 1,
        "new_device": 1,
        "new_location": 1,
        "distance_from_home": 15300.0,
        "transaction_velocity": 7.5,
        "customer_transaction_count": 184,
        "average_transaction_amount": 140.00,
        "transaction_hour": 2,
        "transaction_day": 25,
        "transaction_month": 6,
    }

    result = predict_transaction(high_risk_txn)
    print("\n" + "=" * 60)
    print("REAL-TIME TRANSACTION SCORING ENGINE OUTPUT")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    print("=" * 60 + "\n")

    alert = create_alert(result)
    if alert:
        print("[ALERT] AUTOMATIC FRAUD ALERT RAISED:")
        print(json.dumps(alert, indent=2))
        print("=" * 60 + "\n")

    # 2. Batch Scoring on First 5,000 Test Records
    test_df = pd.read_csv(TEST_FILE, low_memory=False)
    batch_sample = test_df.head(5000)
    scored_batch = predict_batch(batch_sample)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    scored_batch.to_csv(FRAUD_PREDICTIONS_FILE, index=False)
    logger.info("Batch predictions saved to %s", FRAUD_PREDICTIONS_FILE)

    batch_alerts = generate_batch_alerts(scored_batch)
    if not batch_alerts.empty:
        save_alerts(batch_alerts, ALERTS_FILE)

    print(f"Batch Scored: {len(scored_batch):,} transactions")
    print(f"  * HIGH RISK   : {(scored_batch['risk_level'] == 'HIGH').sum():,}")
    print(f"  * MEDIUM RISK : {(scored_batch['risk_level'] == 'MEDIUM').sum():,}")
    print(f"  * LOW RISK    : {(scored_batch['risk_level'] == 'LOW').sum():,}")
    print(f"  * Alerts Raised: {len(batch_alerts):,}")
    return scored_batch


def main():
    parser = argparse.ArgumentParser(description="Intelligent Banking Transaction & Fraud Detection System")
    parser.add_argument(
        "--stage",
        choices=["all", "data", "train", "anomaly", "evaluate", "predict"],
        default="all",
        help="Pipeline stage to execute",
    )
    args = parser.parse_args()

    t_start = time.time()
    logger.info("=" * 60)
    logger.info("FRAUD DETECTION SYSTEM PIPELINE START | Stage: %s", args.stage)
    logger.info("=" * 60)

    try:
        if args.stage in ("all", "data"):
            stage_data()

        if args.stage in ("all", "train"):
            stage_train()

        if args.stage in ("all", "anomaly"):
            stage_anomaly()

        if args.stage in ("all", "evaluate"):
            stage_evaluate()

        if args.stage in ("all", "predict"):
            stage_predict()

        elapsed = time.time() - t_start
        logger.info("=" * 60)
        logger.info("PIPELINE EXECUTION FINISHED SUCCESSFULLY IN %.2f SECONDS", elapsed)
        logger.info("=" * 60)

    except Exception:
        logger.exception("Pipeline failed with exception.")
        sys.exit(1)


if __name__ == "__main__":
    main()
