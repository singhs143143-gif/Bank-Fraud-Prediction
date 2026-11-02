import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from predict import engineer_anomaly_features, load_artifacts, predict_transaction

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Intelligent Banking Fraud Dashboard", page_icon="??", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: #f4f7fb;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
    }
    .kpi-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04);
        height: 100%;
    }
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .dashboard-title {
        font-size: 2.15rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.04em;
    }
    .muted {
        color: #475569;
        font-size: 0.9rem;
    }
    .nav-pill {
        background: #e2e8f0;
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        font-size: 0.78rem;
        color: #0f172a;
        font-weight: 600;
        display: inline-block;
        margin-right: 0.45rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
PRESETS = {
    "Custom transaction": ("TRANSFER", 100_000.0, 100_000.0, 0.0, 0.0, 100_000.0, 100),
    "Example: Routine payment ($85.50)": ("PAYMENT", 85.50, 4500.0, 4414.50, 0.0, 0.0, 145),
    "Example: Account drain transfer ($3,500,000)": ("TRANSFER", 3_500_000.0, 3_500_000.0, 0.0, 0.0, 0.0, 210),
    "Example: High-value cash-out ($450,000)": ("CASH_OUT", 450_000.0, 450_000.0, 0.0, 120_000.0, 570_000.0, 320),
}


def risk_segment(prob, anomaly):
    if prob >= 0.80 or (prob >= 0.30 and anomaly >= 0.65):
        return "Critical"
    if prob >= 0.15 or (prob >= 0.02 and anomaly >= 0.60):
        return "High"
    if prob >= 0.01 or anomaly >= 0.50:
        return "Medium"
    return "Low"


@st.cache_data(show_spinner=False)
def load_dashboard_data(limit: int = 15000) -> pd.DataFrame:
    path = Path("data/processed/test.csv")
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "transaction_timestamp" in df.columns:
        df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"], errors="coerce")
    return df.head(limit).copy()


@st.cache_data(show_spinner=False)
def score_batch_dataframe(df_batch: pd.DataFrame) -> pd.DataFrame:
    art = load_artifacts()
    df_batch = df_batch.copy()
    for col in ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        if col in df_batch.columns:
            df_batch[col] = pd.to_numeric(df_batch[col], errors="coerce").fillna(0)

    X = art["feature_engineer"].transform(df_batch)
    probs = art["model"].predict_proba(X)[:, 1]

    anomaly_features = engineer_anomaly_features(df_batch)
    X_iso = art["iso_scaler"].transform(
        art["iso_imputer"].transform(anomaly_features[art["iso_numeric_features"]])
    )
    raw = -art["iso_model"].score_samples(X_iso)
    span = art["iso_max"] - art["iso_min"] + 1e-9
    anomaly = np.clip((raw - art["iso_min"]) / span, 0.0, 1.0)

    scored = df_batch.copy()
    scored["fraud_probability"] = np.round(probs, 4)
    scored["anomaly_score"] = np.round(anomaly, 4)
    scored["predicted_fraud"] = (probs >= art["threshold"]).astype(int)
    scored["risk_segment"] = [risk_segment(p, a) for p, a in zip(probs, anomaly)]
    return scored


@st.cache_data(show_spinner=False)
def get_model_comparison() -> pd.DataFrame:
    path = Path("reports/model_comparison.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_feature_importance() -> pd.DataFrame:
    path = Path("reports/feature_importance.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = df.copy()
    filtered["transaction_timestamp"] = pd.to_datetime(filtered["transaction_timestamp"], errors="coerce")

    min_date = st.session_state.get("date_start")
    max_date = st.session_state.get("date_end")
    if min_date is not None:
        filtered = filtered[filtered["transaction_timestamp"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        filtered = filtered[filtered["transaction_timestamp"] <= pd.Timestamp(max_date)]

    selected_types = st.session_state.get("txn_type_filter", [])
    if selected_types:
        filtered = filtered[filtered["transaction_type"].isin(selected_types)]

    selected_merchants = st.session_state.get("merchant_filter", [])
    if selected_merchants:
        filtered = filtered[filtered["merchant_category"].isin(selected_merchants)]

    selected_methods = st.session_state.get("payment_filter", [])
    if selected_methods:
        filtered = filtered[filtered["payment_method"].isin(selected_methods)]

    selected_locations = st.session_state.get("location_filter", [])
    if selected_locations:
        filtered = filtered[filtered["transaction_location"].isin(selected_locations)]

    selected_device = st.session_state.get("device_filter", [])
    if selected_device:
        filtered = filtered[filtered["device_type"].isin(selected_device)]

    selected_risk = st.session_state.get("risk_filter", [])
    if selected_risk:
        filtered = filtered[filtered["risk_segment"].isin(selected_risk)]

    fraud_status = st.session_state.get("fraud_status_filter")
    if fraud_status == "Fraud":
        filtered = filtered[filtered["fraud_label"] == 1]
    elif fraud_status == "Legitimate":
        filtered = filtered[filtered["fraud_label"] == 0]

    return filtered.reset_index(drop=True)


def render_overview_page(scored: pd.DataFrame):
    if scored.empty:
        st.info("No transactions available for the selected filters.")
        return

    total_txns = len(scored)
    total_amt = float(scored["transaction_amount"].sum())
    fraud_count = int(scored["fraud_label"].sum()) if "fraud_label" in scored.columns else 0
    fraud_amt = float(scored.loc[scored["fraud_label"] == 1, "transaction_amount"].sum()) if "fraud_label" in scored.columns else 0.0
    fraud_rate = fraud_count / total_txns if total_txns else 0.0
    high_risk = int(((scored["risk_segment"] == "High") | (scored["risk_segment"] == "Critical")).sum())
    total_flagged = int(scored["predicted_fraud"].sum()) if "predicted_fraud" in scored.columns else 0
    false_positive_count = int(((scored["fraud_label"] == 0) & (scored["predicted_fraud"] == 1)).sum()) if "predicted_fraud" in scored.columns and "fraud_label" in scored.columns else 0
    false_positive_rate = false_positive_count / max((scored["fraud_label"] == 0).sum(), 1)

    st.markdown("<div class='section-header'>Executive Fraud Overview</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    metrics = [
        ("Total Transactions", f"{total_txns:,.0f}", "Operational volume", "#0f172a"),
        ("Transaction Value", f"${total_amt:,.0f}", "Gross value in period", "#2563eb"),
        ("Fraud Transactions", f"{fraud_count:,}", f"{fraud_rate:.2%} of volume", "#dc2626"),
        ("Fraud Amount", f"${fraud_amt:,.0f}", "Actual fraud exposure", "#f97316"),
    ]
    for i, (label, value, detail, color) in enumerate(metrics):
        with cols[i]:
            st.markdown(f"<div class='kpi-card'><div class='muted'>{label}</div><div style='font-size:2rem;font-weight:800;color:{color};'>{value}</div><div class='muted'>{detail}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Fraud Activity Trend</div>", unsafe_allow_html=True)
    trend_df = scored.copy()
    if "transaction_timestamp" in trend_df.columns:
        trend_df = trend_df.sort_values("transaction_timestamp")
        trend_df["date"] = trend_df["transaction_timestamp"].dt.floor("D")
        trend = trend_df.groupby("date").agg(fraud_count=("fraud_label", "sum"), total_transactions=("transaction_id", "count")).reset_index()
        st.line_chart(trend.set_index("date"), x="date")

    left, right = st.columns([1.4, 1.0])
    with left:
        st.markdown("<div class='section-header'>Transaction Risk Distribution</div>", unsafe_allow_html=True)
        risk_counts = scored["risk_segment"].value_counts().reindex(["Low", "Medium", "High", "Critical"], fill_value=0)
        st.bar_chart(risk_counts)
    with right:
        st.markdown("<div class='section-header'>Risk Snapshot</div>", unsafe_allow_html=True)
        summary = pd.DataFrame({
            "metric": ["High-Risk Transactions", "Flagged by Model", "False Positive Rate"],
            "value": [high_risk, total_flagged, f"{false_positive_rate:.2%}"]
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

    st.markdown("<div class='section-header'>Fraud by Transaction Type</div>", unsafe_allow_html=True)
    type_df = scored.groupby("transaction_type").agg(
        transactions=("transaction_id", "count"),
        fraud_count=("fraud_label", "sum"),
    ).reset_index()
    type_df["fraud_rate"] = type_df["fraud_count"] / type_df["transactions"]
    type_df = type_df.sort_values("fraud_rate", ascending=False).head(8)
    st.bar_chart(type_df.set_index("transaction_type")["fraud_rate"].mul(100))

    st.markdown("<div class='section-header'>Fraud Hotspots by Location</div>", unsafe_allow_html=True)
    if "transaction_location" in scored.columns:
        loc = scored.groupby("transaction_location").agg(transactions=("transaction_id", "count"), fraud_count=("fraud_label", "sum")).reset_index()
        loc["fraud_rate"] = loc["fraud_count"] / loc["transactions"]
        loc = loc.sort_values("fraud_rate", ascending=False).head(8)
        st.bar_chart(loc.set_index("transaction_location")["fraud_rate"].mul(100))


def render_intelligence_page(scored: pd.DataFrame):
    st.markdown("<div class='section-header'>Fraud Pattern Matrix</div>", unsafe_allow_html=True)
    if "transaction_type" in scored.columns and "merchant_category" in scored.columns:
        pattern = scored.groupby(["transaction_type", "merchant_category"]).agg(
            transactions=("transaction_id", "count"),
            fraud_count=("fraud_label", "sum"),
        ).reset_index()
        pattern["fraud_rate"] = pattern["fraud_count"] / pattern["transactions"]
        st.dataframe(pattern.sort_values("fraud_rate", ascending=False).head(25), hide_index=True, use_container_width=True)

    st.markdown("<div class='section-header'>Behavioral Risk Signals</div>", unsafe_allow_html=True)
    signal_cols = ["new_device", "new_location", "transaction_velocity", "international_transaction", "failed_transaction_count", "night_transaction"]
    rows = []
    for col in signal_cols:
        if col in scored.columns:
            affected = scored[scored[col].astype(float) > 0]
            if not affected.empty:
                rows.append({
                    "signal": col.replace("_", " ").title(),
                    "affected_transactions": len(affected),
                    "fraud_rate": float(affected["fraud_label"].mean()) if "fraud_label" in affected.columns else 0.0,
                    "fraud_contribution": int(affected["fraud_label"].sum()) if "fraud_label" in affected.columns else 0,
                })
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("fraud_rate", ascending=False), hide_index=True, use_container_width=True)


def render_investigation_page(scored: pd.DataFrame):
    if scored.empty:
        st.info("No transaction data available.")
        return

    if "risk_score" not in scored.columns:
        scored["risk_score"] = (scored["fraud_probability"] * 100).round().astype(int)
    if "recommended_action" not in scored.columns:
        scored["recommended_action"] = np.where(scored["risk_segment"].isin(["High", "Critical"]), "REVIEW", "ALLOW")

    cols = [
        "transaction_id", "customer_id", "transaction_timestamp", "transaction_amount",
        "transaction_type", "merchant_category", "transaction_location", "device_type",
        "fraud_probability", "anomaly_score", "risk_score", "risk_segment", "fraud_label", "recommended_action"
    ]
    display_df = scored[cols].copy()
    display_df["transaction_amount"] = display_df["transaction_amount"].map(lambda v: f"${v:,.2f}")
    display_df["fraud_probability"] = display_df["fraud_probability"].map(lambda v: f"{v:.2%}")
    display_df["anomaly_score"] = display_df["anomaly_score"].map(lambda v: f"{v:.2%}")

    st.markdown("<div class='section-header'>Transaction Investigation</div>", unsafe_allow_html=True)
    left, right = st.columns([1.6, 1.0])
    with left:
        st.dataframe(display_df.head(20), hide_index=True, use_container_width=True)
    with right:
        selected_txn = st.selectbox("Select transaction", display_df["transaction_id"].tolist())
        detail = display_df[display_df["transaction_id"] == selected_txn].iloc[0]
        st.write(f"Customer ID: **{detail['customer_id']}**")
        st.write(f"Amount: **{detail['transaction_amount']}**")
        st.write(f"Transaction Type: **{detail['transaction_type']}**")
        st.write(f"Location: **{detail['transaction_location']}**")
        st.write(f"Device: **{detail['device_type']}**")
        st.write(f"Fraud Probability: **{detail['fraud_probability']}**")
        st.write(f"Anomaly Score: **{detail['anomaly_score']}**")
        st.write(f"Risk Level: **{detail['risk_segment']}**")
        st.write(f"Recommended Action: **{detail['recommended_action']}**")


def render_customer_page(scored: pd.DataFrame):
    if "customer_id" not in scored.columns:
        st.info("Customer details are not available in the current dataset.")
        return

    customer_summary = scored.groupby("customer_id").agg(
        transaction_count=("transaction_id", "count"),
        total_amount=("transaction_amount", "sum"),
        fraud_count=("fraud_label", "sum"),
        avg_amount=("transaction_amount", "mean"),
        avg_risk=("fraud_probability", "mean"),
    ).reset_index()
    customer_summary["risk_score"] = (customer_summary["avg_risk"] * 100).round().astype(int)
    customer_summary["risk_level"] = np.where(customer_summary["risk_score"] >= 70, "High", np.where(customer_summary["risk_score"] >= 35, "Medium", "Low"))

    st.markdown("<div class='section-header'>Customer Risk Intelligence</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{customer_summary['customer_id'].nunique():,}")
    c2.metric("High-Risk Customers", f"{(customer_summary['risk_level'] == 'High').sum():,}")
    c3.metric("Customers with Fraud History", f"{(customer_summary['fraud_count'] > 0).sum():,}")
    c4.metric("Avg Risk Score", f"{customer_summary['risk_score'].mean():.1f}")

    st.dataframe(customer_summary.sort_values("risk_score", ascending=False).head(20), hide_index=True, use_container_width=True)


def render_model_page():
    model_df = get_model_comparison()
    if model_df.empty:
        st.warning("No model comparison file found in reports/model_comparison.csv.")
        return

    st.markdown("<div class='section-header'>Model Performance</div>", unsafe_allow_html=True)
    st.dataframe(model_df.sort_values("f1", ascending=False), hide_index=True, use_container_width=True)

    feature_df = get_feature_importance()
    if not feature_df.empty:
        st.markdown("<div class='section-header'>Most Important Features</div>", unsafe_allow_html=True)
        st.bar_chart(feature_df.head(15).set_index("feature")["importance"])


def main():
    st.markdown("<div class='dashboard-title'>Intelligent Banking Transaction & Fraud Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Real-Time Fraud Risk Intelligence & Transaction Monitoring</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:0.7rem;'> <span class='nav-pill'>Overview</span> <span class='nav-pill'>Fraud Intelligence</span> <span class='nav-pill'>Transaction Investigation</span> <span class='nav-pill'>Customer Risk</span> <span class='nav-pill'>Model Performance</span> </div>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("Global Filters")
        base_df = load_dashboard_data()
        if not base_df.empty and "transaction_timestamp" in base_df.columns:
            base_df["transaction_timestamp"] = pd.to_datetime(base_df["transaction_timestamp"], errors="coerce")
            min_dt = base_df["transaction_timestamp"].min()
            max_dt = base_df["transaction_timestamp"].max()
            start_date = st.date_input("Date range start", min_value=min_dt.date(), max_value=max_dt.date(), value=min_dt.date())
            end_date = st.date_input("Date range end", min_value=min_dt.date(), max_value=max_dt.date(), value=max_dt.date())
            st.session_state["date_start"] = pd.Timestamp(start_date)
            st.session_state["date_end"] = pd.Timestamp(end_date)

        if "transaction_type" in base_df.columns:
            st.session_state["txn_type_filter"] = st.multiselect("Transaction type", sorted(base_df["transaction_type"].dropna().unique().tolist()), default=[])
        if "merchant_category" in base_df.columns:
            st.session_state["merchant_filter"] = st.multiselect("Merchant category", sorted(base_df["merchant_category"].dropna().unique().tolist()), default=[])
        if "payment_method" in base_df.columns:
            st.session_state["payment_filter"] = st.multiselect("Payment method", sorted(base_df["payment_method"].dropna().unique().tolist()), default=[])
        if "transaction_location" in base_df.columns:
            st.session_state["location_filter"] = st.multiselect("Location", sorted(base_df["transaction_location"].dropna().unique().tolist()), default=[])
        if "device_type" in base_df.columns:
            st.session_state["device_filter"] = st.multiselect("Device type", sorted(base_df["device_type"].dropna().unique().tolist()), default=[])
        st.session_state["risk_filter"] = st.multiselect("Risk level", ["Low", "Medium", "High", "Critical"], default=[])
        st.session_state["fraud_status_filter"] = st.selectbox("Fraud status", ["All", "Fraud", "Legitimate"])

        if st.button("Reset Filters"):
            for key in list(st.session_state.keys()):
                if key.endswith("_filter") or key.startswith("date_"):
                    del st.session_state[key]

        st.caption("Last data refresh: just now")
        st.caption("Reporting period: filtered view")
        st.caption("Data coverage: production transaction sample")
        st.caption("Model version: XGBoost / Isolation Forest")

    base_df = load_dashboard_data()
    filtered_df = apply_filters(base_df)
    if filtered_df.empty:
        st.warning("The active filter set produced no data. Reset filters to return to the default view.")
        return

    scored = score_batch_dataframe(filtered_df)
    if "fraud_label" in filtered_df.columns:
        scored["fraud_label"] = filtered_df["fraud_label"].reset_index(drop=True)
    if "risk_score" not in scored.columns:
        scored["risk_score"] = (scored["fraud_probability"] * 100).round().astype(int)
    if "predicted_fraud" not in scored.columns:
        scored["predicted_fraud"] = (scored["fraud_probability"] >= 0.4).astype(int)
    if "recommended_action" not in scored.columns:
        scored["recommended_action"] = np.where(scored["risk_segment"].isin(["High", "Critical"]), "REVIEW", "ALLOW")

    overview_tab, intelligence_tab, investigation_tab, customer_tab, model_tab = st.tabs([
        "Overview",
        "Fraud Intelligence",
        "Transaction Investigation",
        "Customer Risk",
        "Model Performance",
    ])

    with overview_tab:
        render_overview_page(scored)
    with intelligence_tab:
        render_intelligence_page(scored)
    with investigation_tab:
        render_investigation_page(scored)
    with customer_tab:
        render_customer_page(scored)
    with model_tab:
        render_model_page()


if __name__ == "__main__":
    main()
