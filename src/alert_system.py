"""
alert_system.py
---------------
Real-time fraud alert triage and life-cycle management module.
Automatically generates actionable investigation alerts when a transaction's
composite risk score exceeds configurable risk thresholds.
"""

import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    ALERT_THRESHOLD,
    ALERTS_FILE,
    LOG_FILE,
    LOG_LEVEL,
    REPORTS_DIR,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger("alert_system")

ALERT_STATUS_CHOICES = ["NEW", "UNDER_REVIEW", "CONFIRMED_FRAUD", "FALSE_POSITIVE", "RESOLVED"]


def create_alert(prediction: Dict, threshold: int = ALERT_THRESHOLD) -> Optional[Dict]:
    """
    Generate an operational alert if risk score meets or exceeds alert threshold.
    """
    risk_score = prediction.get("risk_score", 0)
    if risk_score < threshold:
        return None

    reasons = prediction.get("reasons", [])
    reasons_formatted = "; ".join(reasons) if isinstance(reasons, list) else str(reasons)

    alert = {
        "alert_id": f"ALT_{uuid.uuid4().hex[:8].upper()}",
        "transaction_id": prediction.get("transaction_id", "UNKNOWN"),
        "customer_id": prediction.get("customer_id", "UNKNOWN"),
        "risk_score": int(risk_score),
        "risk_level": prediction.get("risk_level", "HIGH"),
        "fraud_probability": round(float(prediction.get("fraud_probability", 0.0)), 4),
        "anomaly_score": round(float(prediction.get("anomaly_score", 0.0)), 4),
        "alert_reason": reasons_formatted,
        "recommended_action": prediction.get("recommended_action", "INVESTIGATE / BLOCK"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "NEW",
    }
    logger.warning("[FRAUD ALERT] Generated: %s | Txn: %s | Score: %d | Level: %s",
                   alert["alert_id"], alert["transaction_id"], alert["risk_score"], alert["risk_level"])
    return alert


def generate_batch_alerts(scored_df: pd.DataFrame, threshold: int = ALERT_THRESHOLD) -> pd.DataFrame:
    """Filter batch scored predictions and generate operational alerts."""
    high_risk_mask = scored_df["risk_score"] >= threshold
    candidates = scored_df[high_risk_mask].copy()

    if candidates.empty:
        logger.info("No transactions met or exceeded alert threshold (%d).", threshold)
        return pd.DataFrame()

    now_utc = datetime.now(timezone.utc).isoformat()
    alerts = []
    for _, row in candidates.iterrows():
        reasons_raw = row.get("reasons", "")
        reasons_str = "; ".join(reasons_raw) if isinstance(reasons_raw, list) else str(reasons_raw)
        alerts.append({
            "alert_id": f"ALT_{uuid.uuid4().hex[:8].upper()}",
            "transaction_id": row.get("transaction_id", "UNKNOWN"),
            "customer_id": row.get("customer_id", "UNKNOWN"),
            "risk_score": int(row.get("risk_score", 0)),
            "risk_level": row.get("risk_level", "HIGH"),
            "fraud_probability": round(float(row.get("fraud_probability", 0.0)), 4),
            "anomaly_score": round(float(row.get("anomaly_score", 0.0)), 4),
            "alert_reason": reasons_str,
            "recommended_action": row.get("recommended_action", "BLOCK / INVESTIGATE"),
            "created_at": now_utc,
            "status": "NEW",
        })

    alerts_df = pd.DataFrame(alerts)
    logger.info("Generated %d operational alerts from %d transactions (threshold=%d).",
                len(alerts_df), len(scored_df), threshold)
    return alerts_df


def save_alerts(alerts_df: pd.DataFrame, filepath: Path = ALERTS_FILE) -> None:
    """Append alerts to persistent storage."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        existing = pd.read_csv(filepath)
        combined = pd.concat([existing, alerts_df], ignore_index=True)
    else:
        combined = alerts_df

    combined.to_csv(filepath, index=False)
    logger.info("Saved %d total alerts to %s", len(combined), filepath)


if __name__ == "__main__":
    import json
    demo_prediction = {
        "transaction_id": "TXN_ALERT_DEMO",
        "customer_id": "CUST_9901",
        "fraud_probability": 0.945,
        "anomaly_score": 0.82,
        "risk_score": 86,
        "risk_level": "HIGH",
        "recommended_action": "BLOCK / INVESTIGATE",
        "reasons": [
            "Transaction amount ($8,500.00) is 6x customer average",
            "Executed from unrecognized new device",
            "Distance from home: 4,500 km (International location)"
        ]
    }
    print("Testing Alert System independently...")
    alert = create_alert(demo_prediction, threshold=70)
    print("\n[SUCCESS] Generated Operational Alert:")
    print(json.dumps(alert, indent=2))
    print(f"\nSupported Alert Status Lifecycle: {ALERT_STATUS_CHOICES}")
