"""
dashboard_logic.py
------------------
Pure pandas / numpy calculations behind the fraud dashboard (no Streamlit, no plotting).
Keeping the maths here makes every number on the dashboard testable.

Terminology used everywhere in the dashboard:
  * Fraud probability - output of the supervised model (0-1).
  * Anomaly score     - output of the unsupervised Isolation Forest.
  * Risk score        - the combined 0-100 score produced by the prediction pipeline.
  * Confirmed fraud   - the dataset's ground-truth label (fraud_label == 1).
  * Flagged           - the model predicted fraud (probability >= production threshold).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

TIER_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
DEFAULT_CUTOFFS = {"MEDIUM": 31, "HIGH": 61, "CRITICAL": 81}   # 0-30 LOW, 31-60 MEDIUM, 61-80 HIGH, 81-100 CRITICAL
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

STATUS_CONFIRMED = "Confirmed fraud"
STATUS_FLAGGED = "Flagged, not confirmed"
STATUS_LEGIT = "Legitimate"
STATUS_FLAGGED_NOLABEL = "Flagged"
STATUS_NOT_FLAGGED = "Not flagged"


def has(df: pd.DataFrame, *cols: str) -> bool:
    """True if every column exists in the frame."""
    return all(c in df.columns for c in cols)


# -----------------------------------------------------------------------------
# Preparation
# -----------------------------------------------------------------------------
def assign_tier(score: pd.Series, cutoffs: Dict[str, int]) -> pd.Series:
    """Map a 0-100 risk score to LOW / MEDIUM / HIGH / CRITICAL using configurable cut-offs."""
    s = pd.to_numeric(score, errors="coerce").fillna(0).to_numpy()
    tier = np.select(
        [s >= cutoffs["CRITICAL"], s >= cutoffs["HIGH"], s >= cutoffs["MEDIUM"]],
        ["CRITICAL", "HIGH", "MEDIUM"],
        default="LOW",
    )
    return pd.Series(tier, index=score.index)


def prepare(df: pd.DataFrame, cutoffs: Dict[str, int], threshold: float) -> pd.DataFrame:
    """Add the derived columns every page relies on. Never invents source data."""
    out = df.copy()

    out["fraud_probability"] = pd.to_numeric(out["fraud_probability"], errors="coerce").fillna(0.0)
    out["risk_score"] = pd.to_numeric(out["risk_score"], errors="coerce").fillna(0.0)
    if "anomaly_score" in out.columns:
        out["anomaly_score"] = pd.to_numeric(out["anomaly_score"], errors="coerce").fillna(0.0)
    if "transaction_amount" in out.columns:
        out["transaction_amount"] = pd.to_numeric(out["transaction_amount"], errors="coerce").fillna(0.0)

    out["risk_tier"] = assign_tier(out["risk_score"], cutoffs)
    out["flagged"] = out["fraud_probability"] >= threshold

    if "transaction_timestamp" in out.columns:
        out["ts"] = pd.to_datetime(out["transaction_timestamp"], errors="coerce")
    else:
        out["ts"] = pd.NaT

    if "fraud_label" in out.columns:
        out["is_fraud"] = pd.to_numeric(out["fraud_label"], errors="coerce").fillna(0).astype(int)
        out["fraud_status"] = np.where(
            out["is_fraud"] == 1, STATUS_CONFIRMED,
            np.where(out["flagged"], STATUS_FLAGGED, STATUS_LEGIT),
        )
    else:
        out["is_fraud"] = 0
        out["fraud_status"] = np.where(out["flagged"], STATUS_FLAGGED_NOLABEL, STATUS_NOT_FLAGGED)

    if "customer_income" in out.columns and out["customer_income"].nunique() >= 3:
        try:
            out["customer_segment"] = pd.qcut(
                out["customer_income"], 3, labels=["Lower income", "Middle income", "Higher income"], duplicates="drop"
            ).astype(str)
        except ValueError:
            pass
    return out


def add_amount_band(df: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Quantile-based amount bands, so each band holds a similar number of transactions."""
    out = df.copy()
    if "transaction_amount" in out.columns and out["transaction_amount"].nunique() > bins:
        try:
            out["amount_band"] = pd.qcut(out["transaction_amount"], bins, duplicates="drop").astype(str)
        except ValueError:
            pass
    return out


# -----------------------------------------------------------------------------
# Filtering and period comparison
# -----------------------------------------------------------------------------
FILTER_COLUMNS = {
    "transaction_type": "transaction_type",
    "merchant_category": "merchant_category",
    "payment_method": "payment_method",
    "transaction_location": "transaction_location",
    "customer_segment": "customer_segment",
    "risk_tier": "risk_tier",
    "fraud_status": "fraud_status",
    "device_type": "device_type",
}


def apply_categorical_filters(df: pd.DataFrame, selections: Dict[str, List[str]]) -> pd.DataFrame:
    """An empty selection means 'no filter' for that field."""
    mask = pd.Series(True, index=df.index)
    for key, chosen in selections.items():
        col = FILTER_COLUMNS.get(key, key)
        if chosen and col in df.columns:
            mask &= df[col].astype(str).isin([str(c) for c in chosen])
    return df[mask]


def apply_date_filter(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if "ts" not in df.columns or df["ts"].isna().all() or start is None or end is None:
        return df
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) + pd.Timedelta(days=1)
    return df[(df["ts"] >= lo) & (df["ts"] < hi)]


def previous_period(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Rows from the equally long window immediately before [start, end]."""
    if "ts" not in df.columns or df["ts"].isna().all() or start is None or end is None:
        return df.iloc[0:0]
    lo, hi = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    length = hi - lo
    return df[(df["ts"] >= lo - length) & (df["ts"] < lo)]


def pct_change(current: float, previous: float) -> Optional[float]:
    """Percentage change vs previous; None when there is no usable baseline."""
    if previous is None or previous == 0 or pd.isna(previous):
        return None
    return (current - previous) / abs(previous) * 100.0


# -----------------------------------------------------------------------------
# KPIs and classification metrics
# -----------------------------------------------------------------------------
def confusion_counts(df: pd.DataFrame) -> Dict[str, int]:
    fraud = df["is_fraud"] == 1
    flagged = df["flagged"]
    return {
        "tp": int((flagged & fraud).sum()),
        "fp": int((flagged & ~fraud).sum()),
        "fn": int((~flagged & fraud).sum()),
        "tn": int((~flagged & ~fraud).sum()),
    }


def _safe_div(a: float, b: float) -> Optional[float]:
    return None if b == 0 else a / b


def classification_metrics(c: Dict[str, int]) -> Dict[str, Optional[float]]:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = None if precision is None or recall is None or (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": _safe_div(fp, fp + tn),
        "fnr": _safe_div(fn, fn + tp),
    }


def kpis(df: pd.DataFrame, has_label: bool) -> Dict[str, Optional[float]]:
    """Headline numbers. Every rate uses the filtered transaction count as its denominator."""
    n = len(df)
    amount = float(df["transaction_amount"].sum()) if "transaction_amount" in df.columns else 0.0
    fraud = df["is_fraud"] == 1
    out: Dict[str, Optional[float]] = {
        "total": n,
        "value": amount,
        "avg_amount": (amount / n) if n else None,
        "flagged": int(df["flagged"].sum()),
        "high_risk": int(df["risk_tier"].isin(["HIGH", "CRITICAL"]).sum()),
        "high_risk_amount": float(df.loc[df["risk_tier"].isin(["HIGH", "CRITICAL"]), "transaction_amount"].sum())
        if "transaction_amount" in df.columns else 0.0,
        "fraud_txns": int(fraud.sum()) if has_label else None,
        "fraud_amount": float(df.loc[fraud, "transaction_amount"].sum()) if has_label and "transaction_amount" in df.columns else None,
        "fraud_rate": (fraud.sum() / n * 100.0) if has_label and n else None,
        "recall": None,
        "fpr": None,
        "precision": None,
    }
    if has_label and n:
        m = classification_metrics(confusion_counts(df))
        out["recall"] = None if m["recall"] is None else m["recall"] * 100
        out["fpr"] = None if m["fpr"] is None else m["fpr"] * 100
        out["precision"] = None if m["precision"] is None else m["precision"] * 100
    return out


# -----------------------------------------------------------------------------
# Aggregations for charts
# -----------------------------------------------------------------------------
def trend(df: pd.DataFrame, granularity: str = "Daily") -> pd.DataFrame:
    """Per-period counts and amounts. Needs timestamps."""
    if df.empty or df["ts"].isna().all():
        return pd.DataFrame()
    freq = {"Daily": "D", "Weekly": "W", "Monthly": "M"}[granularity]
    d = df.dropna(subset=["ts"]).copy()
    d["period"] = d["ts"].dt.to_period(freq).dt.start_time
    amt = d["transaction_amount"] if "transaction_amount" in d.columns else pd.Series(0.0, index=d.index)
    d["_amt"] = amt
    d["_fraud_amt"] = np.where(d["is_fraud"] == 1, amt, 0.0)
    d["_flag_amt"] = np.where(d["flagged"], amt, 0.0)
    d["_high"] = d["risk_tier"].isin(["HIGH", "CRITICAL"]).astype(int)
    g = d.groupby("period").agg(
        total=("is_fraud", "size"),
        fraud=("is_fraud", "sum"),
        flagged=("flagged", "sum"),
        high_risk=("_high", "sum"),
        amount=("_amt", "sum"),
        fraud_amount=("_fraud_amt", "sum"),
        flagged_amount=("_flag_amt", "sum"),
    ).reset_index()
    g["legitimate"] = g["total"] - g["fraud"]
    g["legitimate_amount"] = g["amount"] - g["fraud_amount"]
    g["fraud_rate"] = np.where(g["total"] > 0, g["fraud"] / g["total"] * 100.0, 0.0)
    return g


def group_fraud(df: pd.DataFrame, col: str, min_n: int = 1) -> pd.DataFrame:
    """Transactions, fraud count, fraud rate and amounts per category value, sorted by fraud rate."""
    if col not in df.columns or df.empty:
        return pd.DataFrame()
    amt = df["transaction_amount"] if "transaction_amount" in df.columns else pd.Series(0.0, index=df.index)
    d = pd.DataFrame({
        "key": df[col].astype(str),
        "fraud": df["is_fraud"],
        "amt": amt,
        "fraud_amt": np.where(df["is_fraud"] == 1, amt, 0.0),
    })
    g = d.groupby("key").agg(
        transactions=("fraud", "size"),
        fraud_txns=("fraud", "sum"),
        volume=("amt", "sum"),
        fraud_amount=("fraud_amt", "sum"),
    ).reset_index().rename(columns={"key": col})
    g = g[g["transactions"] >= min_n].copy()
    g["fraud_rate"] = g["fraud_txns"] / g["transactions"] * 100.0
    return g.sort_values(["fraud_rate", "fraud_txns"], ascending=False).reset_index(drop=True)


def hour_day_heat(df: pd.DataFrame) -> pd.DataFrame:
    """Fraud rate by weekday and hour. Needs timestamps (no guessing the weekday)."""
    if df.empty or df["ts"].isna().all():
        return pd.DataFrame()
    d = df.dropna(subset=["ts"])
    t = pd.DataFrame({
        "day": d["ts"].dt.day_name(),
        "hour": d["ts"].dt.hour,
        "fraud": d["is_fraud"],
    })
    g = t.groupby(["day", "hour"]).agg(transactions=("fraud", "size"), fraud_txns=("fraud", "sum")).reset_index()
    g["fraud_rate"] = g["fraud_txns"] / g["transactions"] * 100.0
    return g


def risk_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Tier counts with the confirmed-fraud count inside each tier (kept separate to avoid double counting)."""
    g = df.groupby("risk_tier").agg(transactions=("is_fraud", "size"), confirmed_fraud=("is_fraud", "sum"))
    g = g.reindex(TIER_ORDER, fill_value=0).reset_index().rename(columns={"risk_tier": "tier"})
    total = g["transactions"].sum()
    g["share"] = np.where(total > 0, g["transactions"] / max(total, 1) * 100.0, 0.0)
    return g


def pattern_matrix(df: pd.DataFrame, row: str, col: str, min_n: int = 10) -> pd.DataFrame:
    """Long-format fraud rate for two categorical dimensions; cells with too few transactions are dropped."""
    if row == col or not has(df, row, col):
        return pd.DataFrame()
    t = pd.DataFrame({"r": df[row].astype(str), "c": df[col].astype(str), "fraud": df["is_fraud"]})
    g = t.groupby(["r", "c"]).agg(transactions=("fraud", "size"), fraud_txns=("fraud", "sum")).reset_index()
    g = g[g["transactions"] >= min_n].copy()
    g["fraud_rate"] = g["fraud_txns"] / g["transactions"] * 100.0
    return g.rename(columns={"r": row, "c": col})


def amount_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / median amount for legitimate vs fraudulent transactions."""
    if "transaction_amount" not in df.columns:
        return pd.DataFrame()
    rows = []
    for label, sub in (("Legitimate", df[df["is_fraud"] == 0]), ("Fraudulent", df[df["is_fraud"] == 1])):
        if len(sub):
            rows.append({"group": label, "transactions": len(sub),
                         "average": float(sub["transaction_amount"].mean()),
                         "median": float(sub["transaction_amount"].median())})
    return pd.DataFrame(rows)


def behavioural_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each suspicious indicator: affected transactions, fraud rate among them, lift vs the
    overall fraud rate, and fraud contribution (share of ALL fraud that carries the signal).
    Only signals whose source columns exist are reported. Thresholds are listed in the 'rule' column.
    """
    signals: Dict[str, tuple] = {}
    if "new_device" in df.columns:
        signals["New device"] = (df["new_device"] == 1, "new_device = 1")
    if "new_location" in df.columns:
        signals["New location"] = (df["new_location"] == 1, "new_location = 1")
    if "transaction_velocity" in df.columns:
        cut = float(df["transaction_velocity"].quantile(0.90))
        signals["High velocity"] = (df["transaction_velocity"] >= cut, f"velocity in top 10% (>= {cut:.1f})")
    if has(df, "transaction_amount", "average_transaction_amount"):
        signals["Unusual amount"] = (
            df["transaction_amount"] > 3 * df["average_transaction_amount"].clip(lower=1e-9),
            "amount > 3x customer average",
        )
    if "international_transaction" in df.columns:
        signals["International transaction"] = (df["international_transaction"] == 1, "international_transaction = 1")
    if "failed_transaction_count" in df.columns:
        signals["Failed authentication"] = (df["failed_transaction_count"] > 0, "failed attempts > 0")
    if "night_transaction" in df.columns:
        signals["Unusual transaction time"] = (df["night_transaction"] == 1, "night_transaction = 1")
    elif "transaction_hour" in df.columns:
        signals["Unusual transaction time"] = (df["transaction_hour"].between(0, 5), "hour between 00:00 and 05:59")

    total_fraud = int((df["is_fraud"] == 1).sum())
    overall = (total_fraud / len(df)) if len(df) else 0.0
    rows = []
    for name, (mask, rule) in signals.items():
        mask = mask.fillna(False)
        n = int(mask.sum())
        f = int(((df["is_fraud"] == 1) & mask).sum())
        rate = (f / n) if n else 0.0
        rows.append({
            "signal": name,
            "affected_transactions": n,
            "fraud_rate": rate * 100.0,
            "lift_vs_overall": (rate / overall) if overall > 0 else np.nan,
            "fraud_contribution": (f / total_fraud * 100.0) if total_fraud else 0.0,
            "rule": rule,
        })
    return pd.DataFrame(rows).sort_values("fraud_rate", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


# -----------------------------------------------------------------------------
# Model evaluation (needs ground-truth labels)
# -----------------------------------------------------------------------------
def _sorted_scores(y: np.ndarray, p: np.ndarray):
    order = np.argsort(-p, kind="mergesort")
    return y[order], p[order]


def roc_curve_points(y: np.ndarray, p: np.ndarray):
    """Returns (fpr, tpr, auc) or None if only one class is present."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return None
    ys, ps = _sorted_scores(y, p)
    tps = np.cumsum(ys)
    fps = np.cumsum(1 - ys)
    distinct = np.where(np.diff(ps))[0]
    idx = np.r_[distinct, len(ys) - 1]
    tpr = np.r_[0.0, tps[idx] / pos]
    fpr = np.r_[0.0, fps[idx] / neg]
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))
    return fpr, tpr, auc


def pr_curve_points(y: np.ndarray, p: np.ndarray):
    """Returns (recall, precision, average_precision) or None if there are no positives."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pos = int(y.sum())
    if pos == 0:
        return None
    ys, ps = _sorted_scores(y, p)
    tps = np.cumsum(ys)
    fps = np.cumsum(1 - ys)
    distinct = np.where(np.diff(ps))[0]
    idx = np.r_[distinct, len(ys) - 1]
    precision = tps[idx] / (tps[idx] + fps[idx])
    recall = tps[idx] / pos
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return np.r_[0.0, recall], np.r_[precision[0], precision], ap


def threshold_sweep(y: np.ndarray, p: np.ndarray, thresholds: Iterable[float]) -> pd.DataFrame:
    """Precision, recall, F1, FP, FN and flagged count at each probability threshold."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=float)
    rows = []
    for t in thresholds:
        flagged = p >= t
        tp = int((flagged & y).sum())
        fp = int((flagged & ~y).sum())
        fn = int((~flagged & y).sum())
        tn = int((~flagged & ~y).sum())
        m = classification_metrics({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
        rows.append({
            "threshold": float(t), "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "false_positives": fp, "false_negatives": fn, "flagged": int(flagged.sum()),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Customers and alerts
# -----------------------------------------------------------------------------
def customer_summary(df: pd.DataFrame, cutoffs: Dict[str, int]) -> pd.DataFrame:
    """
    One row per customer. Customer risk score = the highest transaction risk score for that
    customer (a customer is as risky as their riskiest transaction); tier uses the same cut-offs.
    """
    if "customer_id" not in df.columns or df.empty:
        return pd.DataFrame()
    amt = df["transaction_amount"] if "transaction_amount" in df.columns else pd.Series(0.0, index=df.index)
    d = pd.DataFrame({
        "customer_id": df["customer_id"].astype(str),
        "amt": amt,
        "fraud": df["is_fraud"],
        "anomaly": df["anomaly_score"] if "anomaly_score" in df.columns else 0.0,
        "risk": df["risk_score"],
        "high": df["risk_tier"].isin(["HIGH", "CRITICAL"]).astype(int),
    })
    g = d.groupby("customer_id").agg(
        transactions=("amt", "size"),
        total_amount=("amt", "sum"),
        fraud_count=("fraud", "sum"),
        avg_transaction=("amt", "mean"),
        anomaly_score=("anomaly", "mean"),
        risk_score=("risk", "max"),
        high_risk_txns=("high", "sum"),
    ).reset_index()
    g["risk_tier"] = assign_tier(g["risk_score"], cutoffs).to_numpy()
    return g.sort_values(["risk_score", "fraud_count"], ascending=False).reset_index(drop=True)


def build_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """Alert queue = every HIGH / CRITICAL transaction. Alert ids are derived from transaction ids."""
    if df.empty:
        return pd.DataFrame()
    a = df[df["risk_tier"].isin(["HIGH", "CRITICAL"])].copy()
    if a.empty:
        return pd.DataFrame()
    tid = a["transaction_id"].astype(str) if "transaction_id" in a.columns else a.index.astype(str)
    out = pd.DataFrame({
        "alert_id": "ALT-" + pd.Series(tid, index=a.index).astype(str),
        "transaction_id": pd.Series(tid, index=a.index),
        "customer_id": a["customer_id"].astype(str) if "customer_id" in a.columns else "",
        "risk_score": a["risk_score"],
        "risk_level": a["risk_tier"],
        "alert_reason": a["reasons"].astype(str) if "reasons" in a.columns else "",
        "created_at": a["ts"],
        "recommended_action": a["recommended_action"].astype(str) if "recommended_action" in a.columns else "",
        "is_fraud": a["is_fraud"],
    })
    return out.sort_values("risk_score", ascending=False).reset_index(drop=True)


ALERT_STATUSES = ["New", "Under review", "Confirmed fraud", "False positive", "Resolved"]


# -----------------------------------------------------------------------------
# Formatting
# -----------------------------------------------------------------------------
def fmt_money(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    a = abs(x)
    if a >= 1e9:
        return f"${x / 1e9:,.2f}B"
    if a >= 1e6:
        return f"${x / 1e6:,.2f}M"
    if a >= 1e4:
        return f"${x / 1e3:,.1f}K"
    return f"${x:,.2f}"


def fmt_int(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    x = float(x)
    if abs(x) >= 1e6:
        return f"{x / 1e6:,.2f}M"
    if abs(x) >= 1e4:
        return f"{x / 1e3:,.1f}K"
    return f"{int(round(x)):,}"


def fmt_pct(x: Optional[float], digits: int = 2) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{digits}f}%"


def fmt_delta(x: Optional[float]) -> Optional[str]:
    return None if x is None or pd.isna(x) else f"{x:+.1f}% vs prev. period"