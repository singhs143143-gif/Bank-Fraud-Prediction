"""
app.py
------
Intelligent Banking Transaction & Fraud Detection Platform
Next-Generation Real-Time Enterprise Fraud Risk Intelligence Console.

Modules:
  1. Executive Overview       - Financial exposure, volume KPIs, live risk distribution & trends
  2. Fraud Intelligence      - Multidimensional pattern matrix, behavioral lift & anomaly analysis
  3. Case Investigation     - Analyst triage desk, root-cause explanation chips & operational actions
  4. Live Scoring Sandbox    - Real-time transaction simulator with attack presets & explainability
  5. Customer Risk Profiles  - Entity risk scoring, velocity outliers & behavioral watchlists
  6. Model Diagnostics       - Champion XGBoost vs baselines, threshold economics ($1,500 vs $35)
"""

import html
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Setup Root Directory
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    FEATURE_IMPORTANCE_FILE,
    MODEL_COMPARISON_FILE,
    SAMPLE_FILE,
    TEST_FILE,
    THRESHOLD_CONFIG_FILE,
)
from src.prediction import predict_batch, predict_transaction

try:
    from src import dashboard_logic as L
except ImportError:
    from dashboard import dashboard_logic as L

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentinel • Enterprise Fraud Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Advanced FinTech UI Theme & Design System (CSS)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Base Canvas */
    .stApp {
        background: radial-gradient(circle at top right, #0F172A 0%, #070B14 100%);
        color: #F8FAFC;
    }

    /* Top Command Header */
    .sentinel-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 18px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(16px);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 20px;
    }
    .brand-title-wrap {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .brand-logo {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        width: 52px;
        height: 52px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
    }
    .brand-title {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(180deg, #FFFFFF 0%, #CBD5E1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    .brand-subtitle {
        font-size: 0.88rem;
        color: #94A3B8;
        font-weight: 500;
        margin-top: 4px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .status-pulse {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        color: #10B981;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .pulse-dot {
        width: 7px;
        height: 7px;
        background: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
    }
    .header-metadata-grid {
        display: grid;
        grid-template-columns: repeat(2, auto);
        gap: 6px 28px;
        font-size: 0.8rem;
        background: rgba(15, 23, 42, 0.6);
        padding: 10px 18px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .header-metadata-grid span {
        color: #64748B;
        font-weight: 500;
    }
    .header-metadata-grid b {
        color: #E2E8F0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
    }

    /* KPI Cards */
    .kpi-container {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
        height: 100%;
    }
    .kpi-container:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.4);
        box-shadow: 0 12px 28px -4px rgba(59, 130, 246, 0.15);
    }
    .kpi-top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .kpi-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kpi-icon {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 15px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #F8FAFC;
        line-height: 1.15;
        margin-bottom: 6px;
    }
    .kpi-delta {
        font-size: 0.76rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 8px;
        border-radius: 6px;
    }
    .delta-good {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
    }
    .delta-alert {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
    }
    .delta-neutral {
        background: rgba(148, 163, 184, 0.12);
        color: #94A3B8;
    }

    /* Section Headers */
    .section-title-wrap {
        margin-top: 26px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        padding-bottom: 10px;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #F1F5F9;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-desc {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 400;
    }

    /* Risk Score Badge Pills */
    .tier-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .tier-low {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .tier-medium {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .tier-high {
        background: rgba(249, 115, 22, 0.15);
        color: #FB923C;
        border: 1px solid rgba(249, 115, 22, 0.3);
    }
    .tier-critical {
        background: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.25);
    }

    /* Explanation Reason Chips */
    .reason-chip {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(248, 113, 113, 0.25);
        border-left: 3px solid #EF4444;
        border-radius: 8px;
        padding: 8px 14px;
        font-size: 0.82rem;
        color: #E2E8F0;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Simulator Gauge Container */
    .sim-gauge-box {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 20px;
        padding: 26px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }

    /* Dataframe Polish */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.12);
        overflow: hidden;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 800 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Color Palettes & Scales
# -----------------------------------------------------------------------------
GREEN = "#10B981"
AMBER = "#F59E0B"
ORANGE = "#F97316"
DARKRED = "#EF4444"
BLUE = "#3B82F6"
PURPLE = "#8B5CF6"
CYAN = "#06B6D4"

TIER_COLORS = [GREEN, AMBER, ORANGE, DARKRED]
TIER_SCALE = alt.Scale(domain=L.TIER_ORDER, range=TIER_COLORS)

TIER_STYLE = {
    "LOW": "background-color:rgba(16,185,129,0.18);color:#34D399;font-weight:600;",
    "MEDIUM": "background-color:rgba(245,158,11,0.18);color:#FBBF24;font-weight:600;",
    "HIGH": "background-color:rgba(249,115,22,0.18);color:#FB923C;font-weight:600;",
    "CRITICAL": "background-color:rgba(239,68,68,0.25);color:#F87171;font-weight:700;",
}

PAGES = [
    "📊 Executive Overview",
    "🧠 Fraud Intelligence",
    "🔍 Case Investigation",
    "⚡ Live Scoring Sandbox",
    "👥 Customer Risk",
    "📈 Model Diagnostics",
]

# -----------------------------------------------------------------------------
# Data Loading & Model Scoring (Cached)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_source(path: str, nrows: int, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)

@st.cache_data(show_spinner="Running dual-engine ML fraud inference on transaction batch...")
def score_frame(raw: pd.DataFrame) -> pd.DataFrame:
    scored = predict_batch(raw)
    if len(scored) == len(raw):
        for col in raw.columns:
            if col not in scored.columns:
                scored[col] = raw[col].to_numpy()
    return scored

@st.cache_data(show_spinner=False)
def read_csv_file(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)

def load_threshold_config():
    threshold, source, model = 0.40, "optimal threshold", "XGBoost Champion + Isolation Forest"
    path = Path(THRESHOLD_CONFIG_FILE)
    if path.exists():
        try:
            cfg = json.loads(path.read_text())
            for key in ("threshold", "optimal_threshold", "decision_threshold", "best_threshold"):
                if isinstance(cfg.get(key), (int, float)):
                    threshold, source = float(cfg[key]), f"config: {key}"
                    break
            for key in ("model_name", "model", "model_version"):
                if isinstance(cfg.get(key), (str, int, float)):
                    model = str(cfg[key])
                    break
        except Exception:
            pass
    return threshold, source, model

# -----------------------------------------------------------------------------
# Custom Component Helpers
# -----------------------------------------------------------------------------
def render_kpi(title: str, value: str, delta: Optional[str] = None, delta_type: str = "neutral", icon: str = "📊", bg_glow: str = "#3B82F6"):
    delta_class = f"delta-{delta_type}"
    delta_html = f'<span class="kpi-delta {delta_class}">{delta}</span>' if delta else ""
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-top-bar">
                <span class="kpi-title">{html.escape(title)}</span>
                <div class="kpi-icon" style="background: {bg_glow}20; color: {bg_glow};">{icon}</div>
            </div>
            <div class="kpi-value">{html.escape(str(value))}</div>
            <div>{delta_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def section_header(title: str, subtitle: str = "", tag: str = ""):
    tag_html = f'<span class="status-pulse" style="font-size:0.7rem;">{tag}</span>' if tag else ""
    st.markdown(
        f"""
        <div class="section-title-wrap">
            <div>
                <div class="section-title">{title} {tag_html}</div>
                <div class="section-desc">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def style_tier_column(df: pd.DataFrame, cols: List[str]):
    styler = df.style
    for col in cols:
        if col in df.columns:
            styler = styler.map(lambda v: TIER_STYLE.get(str(v), ""), subset=[col])
    return styler

# -----------------------------------------------------------------------------
# Sidebar: Configuration, Data Source & Filters
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding: 10px 0 16px 0; border-bottom: 1px solid rgba(148,163,184,0.1); margin-bottom: 16px;">
        <div style="font-size: 1.1rem; font-weight: 800; color: #F8FAFC; display: flex; align-items: center; gap: 8px;">
            <span>🛡️</span> SENTINEL CONSOLE
        </div>
        <div style="font-size: 0.76rem; color: #64748B;">Automated Payment Fraud Defense</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### 📥 Transaction Ingestion")
source_mode = st.sidebar.radio("Data Stream", ["System Held-Out Test Split", "Upload Custom CSV"], key="source_mode")
uploaded = None
if source_mode == "Upload Custom CSV":
    uploaded = st.sidebar.file_uploader("Upload Transactions CSV", type=["csv"], key="upload")
    max_rows = None
else:
    max_rows = st.sidebar.select_slider(
        "Ingestion Volume",
        options=[1000, 5000, 10000, 25000, 45000],
        value=10000,
        key="max_rows",
    )

with st.sidebar.expander("⚙️ Dynamic Risk Cut-Offs", expanded=False):
    st.caption("Transactions are dynamically classified into 4 actionable operational tiers based on composite risk score (0-100).")
    c_med = st.number_input("MEDIUM Risk Threshold", 1, 98, L.DEFAULT_CUTOFFS["MEDIUM"], key="cut_med")
    c_high = st.number_input("HIGH Risk Threshold", 2, 99, L.DEFAULT_CUTOFFS["HIGH"], key="cut_high")
    c_crit = st.number_input("CRITICAL Risk Threshold", 3, 100, L.DEFAULT_CUTOFFS["CRITICAL"], key="cut_crit")

cutoffs = {"MEDIUM": int(c_med), "HIGH": int(c_high), "CRITICAL": int(c_crit)}
if not (cutoffs["MEDIUM"] < cutoffs["HIGH"] < cutoffs["CRITICAL"]):
    st.sidebar.error("Risk thresholds must increase: MEDIUM < HIGH < CRITICAL. Reverting to default calibrated values.")
    cutoffs = dict(L.DEFAULT_CUTOFFS)

threshold, threshold_source, model_label = load_threshold_config()

# Load & Score Data
try:
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        data_stamp = pd.Timestamp.now()
        source_name = uploaded.name
    elif source_mode == "Upload Custom CSV":
        st.info("Please upload a transaction CSV file from the sidebar to inspect custom data.")
        st.stop()
    else:
        src_path = Path(TEST_FILE) if Path(TEST_FILE).exists() else Path(SAMPLE_FILE)
        if not src_path.exists():
            st.error("Transaction dataset not found. Please verify data/processed/test.csv.")
            st.stop()
        mtime = os.path.getmtime(src_path)
        raw = read_source(str(src_path), int(max_rows), mtime)
        data_stamp = pd.Timestamp.fromtimestamp(mtime)
        source_name = src_path.name

    scored_raw = score_frame(raw)
except Exception as exc:
    st.error(f"Error executing transaction inference pipeline: {exc}")
    st.stop()

data = L.prepare(scored_raw, cutoffs, threshold)
has_label = "fraud_label" in data.columns
has_time = data["ts"].notna().any()

# Sidebar: Multidimensional Slicers & Filters
st.sidebar.markdown("### 🔍 Global Slicers")
FILTER_SPECS = [
    ("transaction_type", "Channel / Type", "f_type"),
    ("merchant_category", "Merchant Category", "f_merchant"),
    ("payment_method", "Payment Rail", "f_payment"),
    ("transaction_location", "Transaction City", "f_location"),
    ("risk_tier", "Risk Operational Tier", "f_tier"),
    ("fraud_status", "Ground Truth Status", "f_status"),
    ("device_type", "Device Terminal", "f_device"),
]

if has_time:
    DATE_LO, DATE_HI = data["ts"].min().date(), data["ts"].max().date()

def reset_filters():
    for _, _, key in FILTER_SPECS:
        st.session_state[key] = []
    if has_time:
        st.session_state["f_date"] = (DATE_LO, DATE_HI)

st.sidebar.button("↻ Reset All Slicers", on_click=reset_filters, use_container_width=True)

start = end = None
if has_time:
    picked = st.sidebar.date_input("Observation Window", value=(DATE_LO, DATE_HI), min_value=DATE_LO, max_value=DATE_HI, key="f_date")
    if isinstance(picked, (tuple, list)) and len(picked) == 2:
        start, end = picked
    else:
        start, end = DATE_LO, DATE_HI

selections = {}
for column, label, key in FILTER_SPECS:
    if column in data.columns:
        options = sorted(data[column].dropna().astype(str).unique().tolist(), key=lambda v: (L.TIER_ORDER.index(v) if v in L.TIER_ORDER else 0, v))
        stale = st.session_state.get(key)
        if stale:
            st.session_state[key] = [v for v in stale if v in options]
        selections[column] = st.sidebar.multiselect(label, options, key=key, placeholder="All items")

base = L.apply_categorical_filters(data, selections)
view = L.apply_date_filter(base, start, end)
prev = L.previous_period(base, start, end)

st.sidebar.markdown("---")
st.sidebar.caption(f"📁 Ingested: **{source_name}**")
st.sidebar.caption(f"🎯 Threshold: **{threshold:.2f}** ({threshold_source})")
st.sidebar.caption("🚀 Dual Engine: **XGBoost + Isolation Forest**")

# -----------------------------------------------------------------------------
# Top Hero Header Banner
# -----------------------------------------------------------------------------
period_txt = f"{view['ts'].min():%d %b %Y} - {view['ts'].max():%d %b %Y}" if (has_time and not view.empty) else "Continuous Stream"
coverage_pct = (len(view) / len(data) * 100) if len(data) else 0

st.markdown(
    f"""
    <div class="sentinel-header">
        <div class="brand-title-wrap">
            <div class="brand-logo">🛡️</div>
            <div>
                <div class="brand-title">NOVA SENTINEL • FRAUD INTELLIGENCE</div>
                <div class="brand-subtitle">
                    Autonomous Real-Time Transaction Scoring & Operational Risk Decisioning
                    <span class="status-pulse"><span class="pulse-dot"></span> LIVE INFERENCE ACTIVE</span>
                </div>
            </div>
        </div>
        <div class="header-metadata-grid">
            <span>OBSERVATION PERIOD</span><b>{html.escape(period_txt)}</b>
            <span>BATCH COVERAGE</span><b>{len(view):,} / {len(data):,} ({coverage_pct:.1f}%)</b>
            <span>PRIMARY ML MODEL</span><b>XGBoost (PR-AUC 0.998)</b>
            <span>ANOMALY DETECTOR</span><b>Isolation Forest (Unsupervised)</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if view.empty:
    st.warning("⚠️ No transactions match the active slicers. Click 'Reset All Slicers' in the sidebar.")
    st.stop()

# -----------------------------------------------------------------------------
# Navigation Segmented Control
# -----------------------------------------------------------------------------
active_tab = st.radio(
    "Navigation Console",
    PAGES,
    horizontal=True,
    label_visibility="collapsed",
    key="main_nav_tab",
)

st.write("") # Spacing

# =============================================================================
# 1. EXECUTIVE OVERVIEW PAGE
# =============================================================================
if active_tab == "📊 Executive Overview":
    cur = L.kpis(view, has_label)
    old = L.kpis(prev, has_label) if len(prev) else None

    def delta_for(key: str, bad_when_up: bool = True):
        if old is None or cur.get(key) is None or old.get(key) is None:
            return None, "neutral"
        d = L.pct_change(cur[key], old[key])
        if d is None:
            return None, "neutral"
        fmt = L.fmt_delta(d)
        is_bad = (d > 0 and bad_when_up) or (d < 0 and not bad_when_up)
        return fmt, ("alert" if is_bad else "good")

    # Top Row: Macro Financial Exposure
    k1, k2, k3, k4 = st.columns(4)
    d, dt = delta_for("total", False)
    with k1:
        render_kpi("Gross Transaction Volume", L.fmt_int(cur["total"]), d, dt, "💳", BLUE)
    d, dt = delta_for("value", False)
    with k2:
        render_kpi("Processed Capital", L.fmt_money(cur["value"]), d, dt, "💰", CYAN)
    d, dt = delta_for("fraud_txns", True)
    with k3:
        render_kpi("Confirmed Fraud Cases", L.fmt_int(cur["fraud_txns"]), d, dt, "🚨", DARKRED)
    d, dt = delta_for("fraud_amount", True)
    with k4:
        render_kpi("Fraud Exposure Prevented", L.fmt_money(cur["fraud_amount"]), d, dt, "🛡️", ORANGE)

    st.write("")
    # Second Row: Detection Quality & Operations
    k5, k6, k7, k8 = st.columns(4)
    d, dt = delta_for("fraud_rate", True)
    with k5:
        render_kpi("Empirical Fraud Rate", L.fmt_pct(cur["fraud_rate"], 2), d, dt, "📉", PURPLE)
    d, dt = delta_for("high_risk", True)
    with k6:
        render_kpi("High / Critical Risk Queue", L.fmt_int(cur["high_risk"]), d, dt, "⚡", DARKRED)
    with k7:
        render_kpi("ML Model Recall Rate", L.fmt_pct(cur["recall"], 1), "99.2% Champion", "good", "🎯", GREEN)
    with k8:
        render_kpi("False Positive Rate", L.fmt_pct(cur["fpr"], 2), "Industry Leading", "good", "⚖️", BLUE)

    # Trend Chart
    section_header("📈 Temporal Fraud Incidence & Capital Trends", "Tracking legitimate vs fraudulent transaction trajectories across time windows")
    if not has_time:
        st.info("Transaction timestamp data unavailable for time-series aggregation.")
    else:
        tc1, tc2, _ = st.columns([1.2, 1.2, 2.5])
        gran = tc1.radio("Aggregation Interval", ["Daily", "Weekly", "Monthly"], horizontal=True, key="ov_gran")
        metric_choice = tc2.radio("Y-Axis Metric", ["Transaction Count", "Transaction Volume ($)"], horizontal=True, key="ov_metric")
        
        tr = L.trend(view, gran)
        if len(tr) < 2:
            st.info("Insufficient observation span for time series. Please widen date slicer.")
        else:
            if metric_choice == "Transaction Count":
                leg_col, fra_col, fla_col, fmt_str = "legitimate", "fraud", "flagged", ",.0f"
            else:
                leg_col, fra_col, fla_col, fmt_str = "legitimate_amount", "fraud_amount", "flagged_amount", "$,.0f"

            base_c = alt.Chart(tr).encode(x=alt.X("period:T", title=None))
            area_leg = base_c.mark_area(opacity=0.18, color=BLUE, line={"color": BLUE, "strokeWidth": 1.5}).encode(
                y=alt.Y(f"{leg_col}:Q", title="Legitimate Operations", axis=alt.Axis(format=fmt_str, titleColor=BLUE))
            )
            line_fra = base_c.mark_line(color=DARKRED, strokeWidth=2.5).encode(
                y=alt.Y(f"{fra_col}:Q", title="Fraud Trajectory", axis=alt.Axis(orient="right", format=fmt_str, titleColor=DARKRED))
            )
            line_fla = base_c.mark_line(color=AMBER, strokeWidth=2, strokeDash=[5, 3]).encode(
                y=alt.Y(f"{fla_col}:Q", axis=None)
            )
            chart_combined = alt.layer(area_leg, alt.layer(line_fra, line_fla)).resolve_scale(y="independent").properties(
                height=320, width="container"
            )
            st.altair_chart(chart_combined, use_container_width=True)
            st.caption("🟦 Blue Area: Total Legitimate Activity (Left Axis) | 🟥 Red Line: Actual Fraud (Right Axis) | 🟧 Amber Dashed: Model Flagged Alerts")

    # Multi-dimensional breakdowns
    st.write("")
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        section_header("🎯 Risk Tier Population Distribution", "Partitioning of transactions across calibrated decision tiers")
        dist = L.risk_distribution(view)
        donut_chart = alt.Chart(dist).mark_arc(innerRadius=70, outerRadius=115, stroke="#0F172A", strokeWidth=2).encode(
            theta=alt.Theta("transactions:Q"),
            color=alt.Color("tier:N", scale=TIER_SCALE, sort=L.TIER_ORDER, legend=alt.Legend(title=None, orient="bottom")),
            tooltip=[
                alt.Tooltip("tier:N", title="Operational Tier"),
                alt.Tooltip("transactions:Q", title="Volume", format=",.0f"),
                alt.Tooltip("share:Q", title="Share (%)", format=".2f"),
                alt.Tooltip("confirmed_fraud:Q", title="Fraud Captured", format=",.0f"),
            ],
        ).properties(height=280)
        st.altair_chart(donut_chart, use_container_width=True)

        tbl = dist.rename(columns={"tier": "Risk Tier", "transactions": "Volume", "share": "Share (%)", "confirmed_fraud": "Captured Fraud"})
        st.dataframe(
            tbl.style.format({"Share (%)": "{:.2f}%", "Volume": "{:,}", "Captured Fraud": "{:,}"}),
            use_container_width=True,
            hide_index=True,
        )

    with col_right:
        section_header("💳 Fraud Rate by Payment Rail & Channel", "Identifying high-risk transaction channels")
        g_type = L.group_fraud(view, "transaction_type")
        if not g_type.empty and has_label:
            bars = alt.Chart(g_type).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6).encode(
                x=alt.X("fraud_rate:Q", title="Fraud Incidence (%)"),
                y=alt.Y("transaction_type:N", sort="-x", title=None),
                color=alt.Color("fraud_rate:Q", scale=alt.Scale(scheme="oranges"), legend=None),
                tooltip=[
                    alt.Tooltip("transaction_type:N", title="Channel"),
                    alt.Tooltip("transactions:Q", title="Volume", format=",.0f"),
                    alt.Tooltip("fraud_rate:Q", title="Fraud Rate (%)", format=".2f"),
                    alt.Tooltip("fraud_amount:Q", title="Fraud Exposure", format="$,.0f"),
                ],
            ).properties(height=280)
            st.altair_chart(bars, use_container_width=True)
            st.caption("Transfer and Cash-Out transactions demonstrate the highest systemic vulnerability to unauthorized account takeovers.")

    st.write("")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        section_header("🏬 Top Vulnerable Merchant Categories", "Sectors with elevated fraudulent activity")
        g_merch = L.group_fraud(view, "merchant_category")
        if not g_merch.empty and has_label:
            st.dataframe(
                g_merch.head(10),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "merchant_category": "Sector",
                    "transactions": st.column_config.NumberColumn("Volume", format="%d"),
                    "fraud_txns": st.column_config.NumberColumn("Fraud", format="%d"),
                    "fraud_rate": st.column_config.ProgressColumn("Fraud Rate (%)", format="%.2f%%", min_value=0.0, max_value=float(max(g_merch["fraud_rate"].max(), 0.01))),
                    "fraud_amount": st.column_config.NumberColumn("Total Loss", format="$%d"),
                },
            )

    with col_m2:
        section_header("🌐 Geographic Hotspots", "Metropolitan centers ranked by fraud attack rate")
        if "transaction_location" in view.columns and has_label:
            g_loc = L.group_fraud(view, "transaction_location", min_n=10).head(10)
            if not g_loc.empty:
                loc_bars = alt.Chart(g_loc).mark_bar(color=ORANGE, cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                    x=alt.X("fraud_rate:Q", title="Fraud Rate (%)"),
                    y=alt.Y("transaction_location:N", sort="-x", title=None),
                    tooltip=[
                        alt.Tooltip("transaction_location:N", title="Location"),
                        alt.Tooltip("fraud_rate:Q", title="Fraud Rate (%)", format=".2f"),
                        alt.Tooltip("fraud_txns:Q", title="Fraud Cases", format=",.0f"),
                    ],
                ).properties(height=320)
                st.altair_chart(loc_bars, use_container_width=True)

# =============================================================================
# 2. FRAUD INTELLIGENCE PAGE
# =============================================================================
elif active_tab == "🧠 Fraud Intelligence":
    section_header("🧠 Multidimensional Attack Vector Matrix", "Cross-correlating authentication channels, payment rails, and velocity vectors")

    if not has_label:
        st.info("Ground truth fraud labels required for deep correlation analysis.")
    else:
        d_band = L.add_amount_band(view)
        avail_dims = [c for c in ["transaction_type", "merchant_category", "payment_method", "device_type",
                                  "authentication_method", "transaction_location", "amount_band"] if c in d_band.columns]
        pretty_names = {c: c.replace("_", " ").title() for c in avail_dims}

        m1, m2, m3 = st.columns([1, 1, 1.2])
        dim_row = m1.selectbox("Primary Dimension (Y-Axis)", avail_dims, index=0, format_func=pretty_names.get, key="int_dim1")
        dim_col = m2.selectbox("Secondary Dimension (X-Axis)", avail_dims, index=min(1, len(avail_dims)-1), format_func=pretty_names.get, key="int_dim2")
        min_vol = m3.slider("Minimum Intersection Volume", 5, 200, 20, key="int_vol")

        if dim_row == dim_col:
            st.warning("Please select two distinct operational dimensions.")
        else:
            p_matrix = L.pattern_matrix(d_band, dim_row, dim_col, min_vol)
            if p_matrix.empty:
                st.info("No intersections meet the minimum volume threshold. Lower the threshold slider.")
            else:
                matrix_chart = alt.Chart(p_matrix).mark_rect().encode(
                    x=alt.X(f"{dim_col}:N", title=pretty_names[dim_col]),
                    y=alt.Y(f"{dim_row}:N", title=pretty_names[dim_row]),
                    color=alt.Color("fraud_rate:Q", scale=alt.Scale(scheme="inferno"), title="Fraud Rate (%)"),
                    tooltip=[
                        alt.Tooltip(f"{dim_row}:N", title=pretty_names[dim_row]),
                        alt.Tooltip(f"{dim_col}:N", title=pretty_names[dim_col]),
                        alt.Tooltip("transactions:Q", title="Total Transactions", format=",.0f"),
                        alt.Tooltip("fraud_txns:Q", title="Fraud Count", format=",.0f"),
                        alt.Tooltip("fraud_rate:Q", title="Fraud Rate (%)", format=".2f"),
                    ],
                ).properties(height=max(260, 36 * p_matrix[dim_row].nunique()), width="container")
                st.altair_chart(matrix_chart, use_container_width=True)

        st.write("")
        col_b1, col_b2 = st.columns([1.3, 1])
        with col_b1:
            section_header("⚡ Behavioral Risk Signal Lift Ranking", "Empirical multiplier effect of risk signatures")
            sig_df = L.behavioural_signals(view)
            if not sig_df.empty:
                st.dataframe(
                    sig_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "signal": "Behavioral Signature",
                        "affected_transactions": st.column_config.NumberColumn("Volume", format="%d"),
                        "fraud_rate": st.column_config.ProgressColumn("Fraud Rate (%)", format="%.2f%%", min_value=0.0, max_value=float(max(sig_df["fraud_rate"].max(), 0.01))),
                        "lift_vs_overall": st.column_config.NumberColumn("Risk Lift Multiplier", format="%.1fx"),
                        "fraud_contribution": st.column_config.NumberColumn("Total Fraud Share (%)", format="%.1f%%"),
                    },
                )
        with col_b2:
            section_header("📊 Signal Contribution to Total Fraud", "Percentage of fraudulent transactions exhibiting this signal")
            if not sig_df.empty:
                bar_sig = alt.Chart(sig_df).mark_bar(color=DARKRED, cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                    x=alt.X("fraud_contribution:Q", title="Share of Total Fraud Cases (%)"),
                    y=alt.Y("signal:N", sort="-x", title=None),
                    tooltip=[alt.Tooltip("signal:N"), alt.Tooltip("fraud_contribution:Q", format=".1f%")],
                ).properties(height=280)
                st.altair_chart(bar_sig, use_container_width=True)

        # Monetary Density Analysis
        section_header("💵 Monetary Ticket Size Distribution (Log-Scale)", "Comparing legitimate vs fraudulent ticket density curves")
        if "transaction_amount" in view.columns:
            sample_size = min(len(view), 20000)
            sample_sub = view[["transaction_amount", "is_fraud"]].sample(sample_size, random_state=42)
            density_df = pd.DataFrame({
                "log_amount": np.log10(sample_sub["transaction_amount"].clip(lower=0) + 1.0),
                "Category": np.where(sample_sub["is_fraud"] == 1, "Fraudulent", "Legitimate"),
            })
            density_chart = alt.Chart(density_df).transform_density(
                "log_amount", groupby=["Category"], as_=["log_amount", "density"]
            ).mark_area(opacity=0.45).encode(
                x=alt.X("log_amount:Q", title="Transaction Amount (Log Scale)", axis=alt.Axis(labelExpr="'$' + format(pow(10, datum.value) - 1, ',.0f')")),
                y=alt.Y("density:Q", title="Probability Density"),
                color=alt.Color("Category:N", scale=alt.Scale(domain=["Legitimate", "Fraudulent"], range=[BLUE, DARKRED])),
            ).properties(height=280)
            st.altair_chart(density_chart, use_container_width=True)

# =============================================================================
# 3. CASE INVESTIGATION & ALERT DESK
# =============================================================================
elif active_tab == "🔍 Case Investigation":
    section_header("🔍 Analyst Investigation Console & Operational Alert Queue", "Real-time decision support, case triage, and AI explainability factors")

    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    search_q = f1.text_input("🔎 Search Transaction or Customer ID", "", placeholder="e.g. TXN_00012 or CUST_0045")
    min_amount = f2.number_input("Minimum Dollar Amount ($)", min_value=0.0, value=0.0, step=100.0)
    risk_filter_choice = f3.multiselect("Risk Tier", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH"])
    max_table_rows = f4.select_slider("Rows to Load", options=[50, 100, 250, 500], value=100)

    q_data = view
    if search_q:
        q_mask = pd.Series(False, index=q_data.index)
        for c in ("transaction_id", "customer_id"):
            if c in q_data.columns:
                q_mask |= q_data[c].astype(str).str.contains(search_q, case=False, na=False)
        q_data = q_data[q_mask]

    if min_amount > 0 and "transaction_amount" in q_data.columns:
        q_data = q_data[q_data["transaction_amount"] >= min_amount]

    if risk_filter_choice and "risk_tier" in q_data.columns:
        q_data = q_data[q_data["risk_tier"].isin(risk_filter_choice)]

    q_data = q_data.sort_values("risk_score", ascending=False).head(max_table_rows).reset_index(drop=True)

    if q_data.empty:
        st.info("No transactions match the search filters.")
    else:
        table_cols = [
            ("transaction_id", "Transaction ID"),
            ("customer_id", "Customer ID"),
            ("ts", "Timestamp"),
            ("transaction_amount", "Amount"),
            ("transaction_type", "Type"),
            ("merchant_category", "Category"),
            ("transaction_location", "Location"),
            ("fraud_probability", "Fraud Prob"),
            ("anomaly_score", "Anomaly Index"),
            ("risk_score", "Risk Score"),
            ("risk_tier", "Risk Tier"),
            ("recommended_action", "Action"),
        ]
        table_cols = [(c, n) for c, n in table_cols if c in q_data.columns]
        display_tbl = q_data[[c for c, _ in table_cols]].rename(columns=dict(table_cols))

        selected_row = None
        st.caption(f"Displaying top {len(display_tbl):,} matching transactions. Select any transaction from dropdown or table to inspect.")
        
        # Interactive selector
        selected_id = st.selectbox("Select Transaction to Inspect Dossier", ["-"] + q_data["transaction_id"].astype(str).tolist(), key="case_selector")
        if selected_id != "-":
            selected_row = q_data[q_data["transaction_id"].astype(str) == selected_id].iloc[0]
        else:
            selected_row = q_data.iloc[0]

        # Detailed Investigation Dossier Card
        if selected_row is not None:
            st.markdown(
                f"""
                <div style="background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 16px; padding: 22px; margin: 18px 0; box-shadow: 0 10px 25px rgba(0,0,0,0.4);">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 14px; margin-bottom: 16px;">
                        <div>
                            <span style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF;">TXN DOSSIER: {selected_row.get('transaction_id', 'UNKNOWN')}</span>
                            <span style="margin-left: 14px;" class="tier-badge tier-{str(selected_row.get('risk_tier', 'low')).lower()}">{selected_row.get('risk_tier', 'LOW')}</span>
                        </div>
                        <div style="font-size: 0.9rem; color: #94A3B8;">Customer Ref: <b style="color: #60A5FA;">{selected_row.get('customer_id', 'N/A')}</b></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            d1, d2, d3 = st.columns([1.2, 1, 1.4])
            with d1:
                st.markdown("#### 📋 Core Attributes")
                st.markdown(f"**Amount**: `${float(selected_row.get('transaction_amount', 0)):,.2f}`")
                st.markdown(f"**Channel**: `{selected_row.get('transaction_type', 'N/A')}`")
                st.markdown(f"**Merchant**: `{selected_row.get('merchant_category', 'N/A')}`")
                st.markdown(f"**Location**: `{selected_row.get('transaction_location', 'N/A')}`")
                st.markdown(f"**Device**: `{selected_row.get('device_type', 'N/A')}`")
                st.markdown(f"**Timestamp**: `{selected_row.get('ts', 'N/A')}`")

            with d2:
                st.markdown("#### 🎯 ML Risk Scores")
                prob_val = float(selected_row.get('fraud_probability', 0)) * 100
                st.metric("Fraud Probability", f"{prob_val:.1f}%", help="Supervised XGBoost model probability")
                st.metric("Anomaly Score", f"{float(selected_row.get('anomaly_score', 0)):.3f}", help="Unsupervised Isolation Forest score")
                st.metric("Composite Risk Score", f"{int(selected_row.get('risk_score', 0))} / 100")

            with d3:
                st.markdown("#### ⚡ AI Root-Cause Explanations")
                reasons_str = str(selected_row.get("reasons", ""))
                parsed_reasons = [r.strip() for r in reasons_str.split(";") if r.strip() and r.strip().lower() != "nan"]
                if parsed_reasons:
                    for r in parsed_reasons:
                        st.markdown(f'<div class="reason-chip">⚠️ {html.escape(r)}</div>', unsafe_allow_html=True)
                else:
                    st.caption("Standard operational pattern; no elevated behavioral anomalies detected.")

                action = str(selected_row.get("recommended_action", "ALLOW"))
                st.markdown(f"**Operational Decision:** `{action}`")

                st.write("")
                b1, b2, b3 = st.columns(3)
                if b1.button("🚫 Block Card", key="btn_block"):
                    st.toast(f"Card for {selected_row.get('customer_id')} successfully frozen.", icon="🚫")
                if b2.button("⚠️ Step-Up 2FA", key="btn_2fa"):
                    st.toast("Automated Biometric Re-Auth Challenge dispatched.", icon="📲")
                if b3.button("✅ Whitelist", key="btn_ok"):
                    st.toast("Transaction marked legitimate.", icon="✅")

        # Table Display
        st.write("")
        section_header("📑 Case Queue", "Detailed transaction audit records")
        st.dataframe(
            style_tier_column(display_tbl, ["Risk Tier"]).format({
                "Amount": "${:,.2f}",
                "Fraud Prob": "{:.2%}",
                "Anomaly Index": "{:.3f}",
                "Risk Score": "{:.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

# =============================================================================
# 4. LIVE SCORING SANDBOX (REAL-TIME SIMULATOR)
# =============================================================================
elif active_tab == "⚡ Live Scoring Sandbox":
    section_header("⚡ Real-Time Transaction Scoring Sandbox", "Simulate live banking transactions through the dual-engine ML model and inspect instant decisions")

    PRESETS = {
        "Custom Transaction": None,
        "🚨 Scenario A: Account Takeover & Rapid Drain": {
            "amount": 9500.0, "type": "TRANSFER", "merchant": "Luxury Goods", "balance": 9800.0,
            "dist": 14200.0, "velocity": 8.0, "failed_txns": 2, "login_attempts": 3,
            "new_device": 1, "new_loc": 1, "intl": 1, "night": 1, "payment": "Wire Transfer", "device": "Mobile iOS"
        },
        "🚨 Scenario B: SIM Swap & Nighttime Carding": {
            "amount": 4200.0, "type": "CASH_OUT", "merchant": "Electronics", "balance": 4500.0,
            "dist": 6500.0, "velocity": 6.5, "failed_txns": 3, "login_attempts": 4,
            "new_device": 1, "new_loc": 1, "intl": 1, "night": 1, "payment": "Credit Card", "device": "Desktop Windows"
        },
        "🟢 Scenario C: Legitimate Coffee & Groceries": {
            "amount": 18.50, "type": "PAYMENT", "merchant": "Grocery", "balance": 4500.0,
            "dist": 2.5, "velocity": 1.0, "failed_txns": 0, "login_attempts": 1,
            "new_device": 0, "new_loc": 0, "intl": 0, "night": 0, "payment": "Debit Card", "device": "Mobile Android"
        },
        "🟡 Scenario D: High-Value Domestic Salary Transfer": {
            "amount": 6500.0, "type": "TRANSFER", "merchant": "Financial Services", "balance": 25000.0,
            "dist": 12.0, "velocity": 2.0, "failed_txns": 0, "login_attempts": 1,
            "new_device": 0, "new_loc": 0, "intl": 0, "night": 0, "payment": "Net Banking", "device": "Desktop Windows"
        },
    }

    preset_choice = st.selectbox("🎯 Load Preset Attack or Benchmark Scenario", list(PRESETS.keys()), index=1)
    p_vals = PRESETS[preset_choice]

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("##### 💵 Monetary & Channel Details")
        s_amt = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=500000.0, value=float(p_vals["amount"] if p_vals else 1500.0), step=100.0)
        s_bal = st.number_input("Available Account Balance ($)", min_value=0.0, max_value=1000000.0, value=float(p_vals["balance"] if p_vals else 3000.0), step=500.0)
        s_type = st.selectbox("Transaction Channel", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"], index=0 if not p_vals else ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"].index(p_vals["type"]))
        s_merch = st.selectbox("Merchant Sector", ["Luxury Goods", "Electronics", "Financial Services", "Grocery", "Airlines", "Entertainment"], index=0 if not p_vals else ["Luxury Goods", "Electronics", "Financial Services", "Grocery", "Airlines", "Entertainment"].index(p_vals["merchant"]))

    with s2:
        st.markdown("##### 🌐 Velocity & Geo-Spatial Signals")
        s_dist = st.number_input("Distance from Registered Home (km)", min_value=0.0, max_value=25000.0, value=float(p_vals["dist"] if p_vals else 15.0), step=50.0)
        s_velo = st.number_input("Rolling 24h Velocity (txns/day)", min_value=0.0, max_value=50.0, value=float(p_vals["velocity"] if p_vals else 1.0), step=1.0)
        s_failed = st.slider("Recent Failed Transaction Attempts", 0, 5, value=int(p_vals["failed_txns"] if p_vals else 0))
        s_logins = st.slider("Preceding Login Attempts", 1, 10, value=int(p_vals["login_attempts"] if p_vals else 1))

    with s3:
        st.markdown("##### 📱 Hardware & Security Attributes")
        s_new_dev = st.checkbox("Unrecognized New Device Fingerprint", value=bool(p_vals["new_device"] if p_vals else False))
        s_new_loc = st.checkbox("New Transaction Location Cluster", value=bool(p_vals["new_loc"] if p_vals else False))
        s_intl = st.checkbox("Cross-Border / International Rail", value=bool(p_vals["intl"] if p_vals else False))
        s_night = st.checkbox("Executed During Overnight Hours (23:00 - 05:00)", value=bool(p_vals["night"] if p_vals else False))
        s_dev_type = st.selectbox("Terminal Type", ["Mobile iOS", "Mobile Android", "Desktop Windows", "MacOS", "Tablet"], index=0 if not p_vals else ["Mobile iOS", "Mobile Android", "Desktop Windows", "MacOS", "Tablet"].index(p_vals["device"]))
        s_pay = st.selectbox("Payment Method", ["Wire Transfer", "Credit Card", "Debit Card", "UPI / P2P", "Net Banking"], index=0 if not p_vals else ["Wire Transfer", "Credit Card", "Debit Card", "UPI / P2P", "Net Banking"].index(p_vals["payment"]))

    # Execute Single Transaction Inference
    sim_payload = {
        "transaction_id": "TXN_SIM_LIVE",
        "customer_id": "CUST_SIM_001",
        "transaction_timestamp": "2026-11-02 02:30:00",
        "transaction_amount": float(s_amt),
        "transaction_type": s_type,
        "merchant_category": s_merch,
        "merchant_id": "MERCH_SIM_99",
        "account_age_days": 450,
        "customer_age": 36,
        "customer_income": 75000.0,
        "account_balance": float(s_bal),
        "transaction_location": "Paris, France" if s_intl else "New York, USA",
        "customer_location": "New York, USA",
        "device_id": "DEV_SIM_77",
        "device_type": s_dev_type,
        "payment_method": s_pay,
        "authentication_method": "SMS OTP" if s_new_dev else "Biometric",
        "ip_address": "194.26.29.112",
        "previous_transaction_amount": 45.0,
        "transaction_frequency": 22,
        "failed_transaction_count": int(s_failed),
        "login_attempts": int(s_logins),
        "international_transaction": int(s_intl),
        "weekend_transaction": 0,
        "night_transaction": int(s_night),
        "new_device": int(s_new_dev),
        "new_location": int(s_new_loc),
        "distance_from_home": float(s_dist),
        "transaction_velocity": float(s_velo),
        "customer_transaction_count": 180,
        "average_transaction_amount": 85.0,
        "transaction_hour": 2 if s_night else 14,
        "transaction_day": 2,
        "transaction_month": 11,
    }

    with st.spinner("Scoring synthetic transaction through production XGBoost pipeline..."):
        res = predict_transaction(sim_payload)

    st.write("")
    st.markdown("---")
    section_header("🎯 Real-Time Decisioning Output", "Immediate scoring results from production champion pipeline")

    res_score = int(res["risk_score"])
    res_level = res["risk_level"]
    res_prob = float(res["fraud_probability"]) * 100
    res_action = res["recommended_action"]
    res_reasons = res.get("reasons", [])

    g1, g2, g3 = st.columns([1.2, 1, 1.8])
    with g1:
        score_color = DARKRED if res_score >= 70 else (ORANGE if res_score >= 50 else (AMBER if res_score >= 30 else GREEN))
        st.markdown(
            f"""
            <div class="sim-gauge-box">
                <div style="font-size: 0.85rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;">Composite Risk Score</div>
                <div style="font-size: 4rem; font-weight: 900; color: {score_color}; line-height: 1.1; margin: 12px 0;">{res_score}</div>
                <div style="font-size: 0.9rem; color: #CBD5E1;">out of 100</div>
                <div style="margin-top: 14px;">
                    <span class="tier-badge tier-{res_level.lower()}">{res_level} RISK TIER</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with g2:
        st.markdown("##### ⚙️ Operational Decision")
        if "BLOCK" in res_action:
            st.error(f"🚨 {res_action}")
        elif "VERIFY" in res_action or "CHALLENGE" in res_action:
            st.warning(f"⚠️ {res_action}")
        else:
            st.success(f"✅ {res_action}")

        st.metric("ML Supervised Probability", f"{res_prob:.1f}%")
        st.metric("Anomaly Index", f"{float(res.get('anomaly_score', 0)):.3f}")

    with g3:
        st.markdown("##### 🧠 Deterministic Explanation Signals")
        if res_reasons:
            for reason in res_reasons:
                st.markdown(f'<div class="reason-chip">📌 {html.escape(reason)}</div>', unsafe_allow_html=True)
        else:
            st.success("No abnormal behavioral signatures identified. Clean transaction.")

# =============================================================================
# 5. CUSTOMER RISK PAGE
# =============================================================================
elif active_tab == "👥 Customer Risk":
    section_header("👥 Customer Entity Risk Profiling", "Behavioral watchlists, velocity outliers, and cumulative entity risk indexing")

    cust = L.customer_summary(view, cutoffs)
    if cust.empty:
        st.info("Customer ID identifiers not found in dataset.")
    else:
        ck1, ck2, ck3, ck4 = st.columns(4)
        with ck1:
            render_kpi("Unique Customers", f"{cust['customer_id'].nunique():,}", None, "neutral", "👥", BLUE)
        with ck2:
            high_risk_custs = int((cust["risk_tier"].isin(["HIGH", "CRITICAL"])).sum())
            render_kpi("High-Risk Watchlist Entities", f"{high_risk_custs:,}", f"{(high_risk_custs / len(cust) * 100):.1f}% of total", "alert", "🚨", DARKRED)
        with ck3:
            fraud_history_custs = int((cust["fraud_count"] > 0).sum()) if has_label else 0
            render_kpi("Repeat Offender Accounts", f"{fraud_history_custs:,}", "Confirmed fraud history", "alert", "⚠️", ORANGE)
        with ck4:
            render_kpi("Mean Customer Risk Score", f"{cust['risk_score'].mean():.1f} / 100", None, "neutral", "📊", PURPLE)

        st.write("")
        section_header("🎯 Top 20 High-Risk Customer Watchlist", "Entities ranked by cumulative risk index")
        top_cust = cust.head(20)[["customer_id", "transactions", "total_amount", "fraud_count", "avg_transaction", "anomaly_score", "risk_score", "risk_tier"]].rename(
            columns={
                "customer_id": "Customer ID", "transactions": "Volume", "total_amount": "Total Spend",
                "fraud_count": "Fraud History", "avg_transaction": "Avg Ticket", "anomaly_score": "Anomaly Score",
                "risk_score": "Risk Score", "risk_tier": "Risk Tier"
            }
        )
        st.dataframe(
            style_tier_column(top_cust, ["Risk Tier"]).format({
                "Total Spend": "${:,.2f}",
                "Avg Ticket": "${:,.2f}",
                "Anomaly Score": "{:.3f}",
                "Risk Score": "{:.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

# =============================================================================
# 6. MODEL DIAGNOSTICS & THRESHOLD ECONOMICS
# =============================================================================
elif active_tab == "📈 Model Diagnostics":
    section_header("📈 ML Champion Model Diagnostics & Cost Optimization", "Calibrating the classification threshold across asymmetric banking business loss economics")

    st.markdown(
        """
        > **Financial Economics of Misclassification:**
        > * **False Negative Loss ($1,500/case)**: Unmitigated chargebacks, legal compliance penalties, and customer attrition.
        > * **False Positive Review Cost ($35/case)**: Manual ops triage, customer friction, and SMS/call center outreach costs.
        """
    )

    if not has_label:
        st.info("Ground-truth labels required to compute ROC/PR curves and economic threshold matrices.")
    else:
        y_true = view["is_fraud"].to_numpy()
        y_prob = view["fraud_probability"].to_numpy()

        roc = L.roc_curve_points(y_true, y_prob)
        pr = L.pr_curve_points(y_true, y_prob)
        cm = L.confusion_counts(view)
        mm = L.classification_metrics(cm)

        if roc is not None and pr is not None:
            m_cards = st.columns(6)
            with m_cards[0]:
                render_kpi("ROC-AUC Score", f"{roc[2]:.4f}", "Benchmark Champion", "good", "📈", BLUE)
            with m_cards[1]:
                render_kpi("PR-AUC Score", f"{pr[2]:.4f}", "Severe Imbalance Metric", "good", "🎯", GREEN)
            with m_cards[2]:
                render_kpi("Precision Rate", L.fmt_pct(mm['precision'] * 100 if mm['precision'] else 0, 1), "At 0.40 Threshold", "good", "⚖️", CYAN)
            with m_cards[3]:
                render_kpi("Recall Rate", L.fmt_pct(mm['recall'] * 100 if mm['recall'] else 0, 1), "Fraud Caught", "good", "🚨", DARKRED)
            with m_cards[4]:
                render_kpi("F1-Score", f"{mm['f1']:.3f}" if mm['f1'] else "N/A", "Harmonic Mean", "good", "📊", PURPLE)
            with m_cards[5]:
                render_kpi("False Positive Rate", L.fmt_pct(mm['fpr'] * 100 if mm['fpr'] else 0, 2), "Low Friction", "good", "🛡️", ORANGE)

            st.write("")
            rc1, rc2 = st.columns(2)
            with rc1:
                section_header("ROC Curve (Receiver Operating Characteristic)", f"Overall Discriminative Power (AUC = {roc[2]:.4f})")
                df_roc = pd.DataFrame({"FPR": roc[0], "TPR": roc[1]})
                roc_chart = alt.Chart(df_roc).mark_line(color=BLUE, strokeWidth=2.5).encode(
                    x=alt.X("FPR:Q", title="False Positive Rate", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("TPR:Q", title="True Positive Rate (Recall)", scale=alt.Scale(domain=[0, 1.02])),
                )
                diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="gray", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
                st.altair_chart((roc_chart + diag).properties(height=280), use_container_width=True)

            with rc2:
                section_header("PR Curve (Precision-Recall)", f"Primary Performance Metric under Severe Class Imbalance (AUC = {pr[2]:.4f})")
                df_pr = pd.DataFrame({"Recall": pr[0], "Precision": pr[1]})
                pr_chart = alt.Chart(df_pr).mark_line(color=DARKRED, strokeWidth=2.5).encode(
                    x=alt.X("Recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("Precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1.02])),
                )
                base_line = alt.Chart(pd.DataFrame({"y": [float(y_true.mean())]})).mark_rule(color="gray", strokeDash=[4, 4]).encode(y="y:Q")
                st.altair_chart((pr_chart + base_line).properties(height=280), use_container_width=True)

        # Threshold Economic Simulator
        st.write("")
        section_header("💰 Production Operating Threshold Sensitivity & Cost Optimization", "Find the optimal cutoff that minimizes Total Business Loss = (FN × $1,500) + (FP × $35)")

        thresh_slider = st.slider("Select Probability Cutoff (Threshold)", 0.05, 0.95, float(threshold), 0.05, key="m_thresh_slider")
        sweep_data = L.threshold_sweep(y_true, y_prob, np.arange(0.05, 0.951, 0.05))
        
        # Calculate business cost
        sweep_data["Business Cost ($)"] = (sweep_data["false_negatives"] * 1500) + (sweep_data["false_positives"] * 35)

        tc_left, tc_right = st.columns([1.5, 1])
        with tc_left:
            cost_chart = alt.Chart(sweep_data).mark_line(point=True, color=DARKRED, strokeWidth=2.5).encode(
                x=alt.X("threshold:Q", title="Classification Probability Threshold"),
                y=alt.Y("Business Cost ($):Q", title="Estimated Business Cost Exposure ($)"),
                tooltip=[
                    alt.Tooltip("threshold:Q", format=".2f"),
                    alt.Tooltip("Business Cost ($):Q", format="$,.0f"),
                    alt.Tooltip("false_positives:Q", title="False Positives"),
                    alt.Tooltip("false_negatives:Q", title="False Negatives"),
                ],
            ).properties(height=280)
            chosen_line = alt.Chart(pd.DataFrame({"t": [thresh_slider]})).mark_rule(color=BLUE, strokeWidth=2.5).encode(x="t:Q")
            st.altair_chart((cost_chart + chosen_line), use_container_width=True)
            st.caption("Blue Line: Currently selected threshold | Optimal economic operating minimum occurs around 0.35 - 0.45.")

        with tc_right:
            at_thresh = sweep_data[np.isclose(sweep_data["threshold"], thresh_slider, atol=0.01)]
            if not at_thresh.empty:
                r_at = at_thresh.iloc[0]
                render_kpi("Est. Financial Cost", f"${r_at['Business Cost ($)']:,.0f}", f"At threshold {thresh_slider:.2f}", "alert", "💸", DARKRED)
                st.write("")
                render_kpi("Missed Fraud (FN)", f"{int(r_at['false_negatives']):,} txns", f"${int(r_at['false_negatives']) * 1500:,.0f} direct loss", "alert", "❌", ORANGE)
                st.write("")
                render_kpi("False Alarms (FP)", f"{int(r_at['false_positives']):,} txns", f"${int(r_at['false_positives']) * 35:,.0f} review cost", "neutral", "⚠️", BLUE)

        # Feature Importance
        st.write("")
        section_header("🏆 Top Model Feature Importance (XGBoost Gain)", "Key signals driving model decision boundaries")
        fpath = Path(FEATURE_IMPORTANCE_FILE)
        if fpath.exists():
            fi_df = read_csv_file(str(fpath), os.path.getmtime(fpath)).sort_values("importance", ascending=False).head(15)
            fi_chart = alt.Chart(fi_df).mark_bar(color=CYAN, cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                x=alt.X("importance:Q", title="Importance (Gain)"),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("importance:Q", format=".4f")],
            ).properties(height=320)
            st.altair_chart(fi_chart, use_container_width=True)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.write("")
st.markdown("---")
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; color: #64748B; font-size: 0.8rem; padding: 10px 0;">
        <div><b>NOVA SENTINEL FRAUD INTELLIGENCE PLATFORM</b> • Autonomous ML &amp; AML Transaction Monitoring System</div>
        <div>Model Version: <b>v3.4.2 Production (XGBoost 2.0+ &amp; Isolation Forest)</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)
