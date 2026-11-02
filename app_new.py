"""
app.py
------
Intelligent Banking Transaction & Fraud Detection - Fraud Risk Intelligence dashboard.

Pages: Overview | Fraud Intelligence | Transaction Investigation | Customer Risk | Model Performance

Every number is computed from the scored transactions (src.prediction.predict_batch output),
the model comparison file, the feature-importance file and the threshold config. Nothing is
hard-coded. Visuals whose source columns are missing say so instead of inventing data.
"""

import html
import json
import os
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

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
from src.prediction import predict_batch
try:
    from src import dashboard_logic as L
except ImportError:
    from dashboard import dashboard_logic as L

st.set_page_config(
    page_title="Fraud Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Style: only the header band uses fixed colours (text and background are both fixed, so it
# reads the same in light and dark mode). Everything else follows the system theme.
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
    .hdr { background:#0B1F3A; color:#FFFFFF; padding:18px 24px; border-radius:10px;
           display:flex; justify-content:space-between; align-items:center; gap:24px;
           flex-wrap:wrap; margin-bottom:12px; }
    .hdr .ttl { font-size:1.45rem; font-weight:700; line-height:1.2; color:#FFFFFF; }
    .hdr .sub { font-size:0.9rem; color:#A9B8D0; margin-top:3px; }
    .hdr .meta { display:grid; grid-template-columns:repeat(2, auto); gap:3px 26px;
                 font-size:0.78rem; color:#A9B8D0; }
    .hdr .meta b { color:#FFFFFF; font-weight:600; }
    div[data-testid="stMetricValue"] { font-size:1.55rem; }
    div[data-testid="stMetricLabel"] p { font-size:0.8rem; }
</style>
""",
    unsafe_allow_html=True,
)

GREEN, RED, ORANGE, AMBER, BLUE, DARKRED = "#2E9E6B", "#D64545", "#E8903A", "#E3B341", "#2F5DA8", "#B3261E"
TIER_COLORS = [GREEN, AMBER, ORANGE, DARKRED]
TIER_SCALE = alt.Scale(domain=L.TIER_ORDER, range=TIER_COLORS)
TIER_STYLE = {
    "LOW": "background-color:#D9F2E4;color:#14532D",
    "MEDIUM": "background-color:#FFF1C9;color:#7A5200",
    "HIGH": "background-color:#FFDCC2;color:#8A3B00",
    "CRITICAL": "background-color:#F8C9C6;color:#7A1C1C;font-weight:700",
}
PAGES = ["Overview", "Fraud Intelligence", "Transaction Investigation", "Customer Risk", "Model Performance"]


# -----------------------------------------------------------------------------
# Data loading (cached)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_source(path: str, nrows: int, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


@st.cache_data(show_spinner="Scoring transactions with the fraud model...")
def score_frame(raw: pd.DataFrame) -> pd.DataFrame:
    scored = predict_batch(raw)
    if len(scored) == len(raw):                      # keep every original column next to the model outputs
        for col in raw.columns:
            if col not in scored.columns:
                scored[col] = raw[col].to_numpy()
    return scored


@st.cache_data(show_spinner=False)
def read_csv_file(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def load_threshold_config():
    """Production threshold and model name from the config file, if it records them."""
    threshold, source, model = 0.5, "default (no threshold in config)", None
    path = Path(THRESHOLD_CONFIG_FILE)
    if path.exists():
        try:
            cfg = json.loads(path.read_text())
            for key in ("threshold", "fraud_threshold", "decision_threshold", "best_threshold", "optimal_threshold"):
                if isinstance(cfg.get(key), (int, float)):
                    threshold, source = float(cfg[key]), f"config: {key}"
                    break
            for key in ("model_name", "model", "model_version", "version"):
                if isinstance(cfg.get(key), (str, int, float)):
                    model = str(cfg[key])
                    break
        except Exception:
            pass
    return threshold, source, model


def metric_card(label, value, delta=None, delta_color="normal", help_text=None):
    with st.container(border=True):
        st.metric(label, value, delta=delta, delta_color=delta_color, help=help_text)


def show(chart):
    st.altair_chart(chart)


def note(text):
    st.caption(text)


# -----------------------------------------------------------------------------
# Header placeholder (filled once filters are known) and navigation
# -----------------------------------------------------------------------------
header_box = st.container()
page = st.radio("Navigation", PAGES, horizontal=True, label_visibility="collapsed", key="page")

# -----------------------------------------------------------------------------
# Sidebar: data source and risk tiers
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Data")
source_mode = st.sidebar.radio("Source", ["System test dataset", "Upload CSV"], key="source_mode")
uploaded = None
if source_mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Transactions CSV", type=["csv"], key="upload")
    max_rows = None
else:
    max_rows = st.sidebar.select_slider("Transactions to analyse", options=[1000, 5000, 10000, 25000, 45000], value=10000, key="max_rows")

with st.sidebar.expander("Risk score tiers", expanded=False):
    st.caption("A transaction's risk score (0-100) falls into the tier whose cut-off it reaches.")
    c_med = st.number_input("MEDIUM from", 1, 98, L.DEFAULT_CUTOFFS["MEDIUM"], key="cut_med")
    c_high = st.number_input("HIGH from", 2, 99, L.DEFAULT_CUTOFFS["HIGH"], key="cut_high")
    c_crit = st.number_input("CRITICAL from", 3, 100, L.DEFAULT_CUTOFFS["CRITICAL"], key="cut_crit")
cutoffs = {"MEDIUM": int(c_med), "HIGH": int(c_high), "CRITICAL": int(c_crit)}
if not (cutoffs["MEDIUM"] < cutoffs["HIGH"] < cutoffs["CRITICAL"]):
    st.sidebar.error("Tier cut-offs must increase (MEDIUM < HIGH < CRITICAL). Using defaults.")
    cutoffs = dict(L.DEFAULT_CUTOFFS)

threshold, threshold_source, model_label = load_threshold_config()

# -----------------------------------------------------------------------------
# Load and score
# -----------------------------------------------------------------------------
try:
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        data_stamp = pd.Timestamp.now()
        source_name = uploaded.name
    elif source_mode == "Upload CSV":
        st.info("Upload a transactions CSV in the sidebar, or switch the source to the system test dataset.")
        st.stop()
    else:
        src_path = Path(TEST_FILE) if Path(TEST_FILE).exists() else Path(SAMPLE_FILE)
        if not src_path.exists():
            st.error("No transaction data found. Check TEST_FILE / SAMPLE_FILE in src/config.py.")
            st.stop()
        mtime = os.path.getmtime(src_path)
        raw = read_source(str(src_path), int(max_rows), mtime)
        data_stamp = pd.Timestamp.fromtimestamp(mtime)
        source_name = src_path.name
    scored_raw = score_frame(raw)
    missing = [c for c in ("fraud_probability", "risk_score") if c not in scored_raw.columns]
    if missing:
        st.error(f"The prediction output is missing required columns: {missing}")
        st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load or score the data: {exc}")
    st.stop()

data = L.prepare(scored_raw, cutoffs, threshold)
has_label = "fraud_label" in data.columns
has_time = data["ts"].notna().any()

# -----------------------------------------------------------------------------
# Sidebar: global filters
# -----------------------------------------------------------------------------
FILTER_SPECS = [
    ("transaction_type", "Transaction type", "f_type"),
    ("merchant_category", "Merchant category", "f_merchant"),
    ("payment_method", "Payment method", "f_payment"),
    ("transaction_location", "Location", "f_location"),
    ("customer_segment", "Customer segment (income band)", "f_segment"),
    ("risk_tier", "Risk level", "f_tier"),
    ("fraud_status", "Fraud status", "f_status"),
    ("device_type", "Device type", "f_device"),
]
if has_time:
    DATE_LO, DATE_HI = data["ts"].min().date(), data["ts"].max().date()


def reset_filters():
    for _, _, key in FILTER_SPECS:
        st.session_state[key] = []
    if has_time:
        st.session_state["f_date"] = (DATE_LO, DATE_HI)


st.sidebar.markdown("### Filters")
st.sidebar.button("Reset filters", on_click=reset_filters)

start = end = None
if has_time:
    picked = st.sidebar.date_input("Date range", value=(DATE_LO, DATE_HI), min_value=DATE_LO, max_value=DATE_HI, key="f_date")
    if isinstance(picked, (tuple, list)) and len(picked) == 2:
        start, end = picked
    else:
        start, end = DATE_LO, DATE_HI

selections = {}
for column, label, key in FILTER_SPECS:
    if column in data.columns:
        options = sorted(data[column].dropna().astype(str).unique().tolist(), key=lambda v: (L.TIER_ORDER.index(v) if v in L.TIER_ORDER else 0, v))
        stale = st.session_state.get(key)
        if stale:                                    # drop choices that no longer exist (e.g. after changing tier cut-offs)
            st.session_state[key] = [v for v in stale if v in options]
        selections[column] = st.sidebar.multiselect(label, options, key=key, placeholder="All")

base = L.apply_categorical_filters(data, selections)
view = L.apply_date_filter(base, start, end)
prev = L.previous_period(base, start, end)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
if has_time and not view.empty:
    period_txt = f"{view['ts'].min():%d %b %Y} to {view['ts'].max():%d %b %Y}"
elif has_time:
    period_txt = "no data in range"
else:
    period_txt = "no timestamps in data"
coverage = f"{len(view):,} of {len(data):,} transactions ({(len(view) / len(data) * 100 if len(data) else 0):.0f}%)"
model_txt = model_label if model_label else "not recorded in config"
with header_box:
    st.markdown(
        f"""
<div class="hdr">
  <div>
    <div class="ttl">Intelligent Banking Transaction &amp; Fraud Detection</div>
    <div class="sub">Real-Time Fraud Risk Intelligence &amp; Transaction Monitoring</div>
  </div>
  <div class="meta">
    <span>Last data refresh</span><b>{html.escape(f"{data_stamp:%d %b %Y, %H:%M}")}</b>
    <span>Reporting period</span><b>{html.escape(period_txt)}</b>
    <span>Data coverage</span><b>{html.escape(coverage)}</b>
    <span>Model</span><b>{html.escape(model_txt)}</b>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

if view.empty:
    st.warning("No transactions match the current filters. Use **Reset filters** in the sidebar.")
    st.stop()

st.sidebar.caption(f"Source: {source_name}")
st.sidebar.caption(f"Production threshold: {threshold:.2f} ({threshold_source})")
if not has_label:
    st.sidebar.warning("No fraud_label column: fraud rate, detection rate and model metrics are unavailable.")


# =============================================================================
# Chart helpers
# =============================================================================
def trend_chart(tr: pd.DataFrame, metric: str) -> alt.Chart:
    if metric == "Transaction count":
        leg, fra, fla, fmt = "legitimate", "fraud", "flagged", ",.0f"
    else:
        leg, fra, fla, fmt = "legitimate_amount", "fraud_amount", "flagged_amount", "$,.0f"
    base_c = alt.Chart(tr).encode(x=alt.X("period:T", title=None))
    leg_l = base_c.mark_area(opacity=0.25, color=GREEN, line={"color": GREEN}).encode(
        y=alt.Y(f"{leg}:Q", title="Legitimate", axis=alt.Axis(format=fmt, titleColor=GREEN))
    )
    fra_l = base_c.mark_line(color=RED, strokeWidth=2.2).encode(
        y=alt.Y(f"{fra}:Q", title="Fraudulent / flagged", axis=alt.Axis(orient="right", format=fmt, titleColor=RED))
    )
    fla_l = base_c.mark_line(color=ORANGE, strokeWidth=2, strokeDash=[5, 3]).encode(y=alt.Y(f"{fla}:Q", axis=None))
    nearest = alt.selection_point(nearest=True, on="mouseover", fields=["period"], empty=False)
    tips = [
        alt.Tooltip("period:T", title="Date"),
        alt.Tooltip("total:Q", title="Total transactions", format=",.0f"),
        alt.Tooltip("fraud:Q", title="Fraud transactions", format=",.0f"),
        alt.Tooltip("fraud_rate:Q", title="Fraud rate (%)", format=".2f"),
        alt.Tooltip("fraud_amount:Q", title="Fraud amount", format="$,.0f"),
        alt.Tooltip("high_risk:Q", title="High-risk transactions", format=",.0f"),
    ]
    rules = base_c.mark_rule(color="gray").encode(
        opacity=alt.condition(nearest, alt.value(0.45), alt.value(0)), tooltip=tips
    ).add_params(nearest)
    return alt.layer(leg_l, alt.layer(fra_l, fla_l), rules).resolve_scale(y="independent").properties(width="container", height=300)


def rank_bars(g: pd.DataFrame, cat: str, title_x: str = "Fraud rate (%)", height: int = 260) -> alt.Chart:
    bars = alt.Chart(g).mark_bar().encode(
        x=alt.X("fraud_rate:Q", title=title_x),
        y=alt.Y(f"{cat}:N", sort=alt.EncodingSortField(field="fraud_rate", order="descending"), title=None),
        color=alt.Color("fraud_rate:Q", scale=alt.Scale(scheme="oranges"), legend=None),
        tooltip=[
            alt.Tooltip(f"{cat}:N", title=cat.replace("_", " ").title()),
            alt.Tooltip("transactions:Q", title="Transactions", format=",.0f"),
            alt.Tooltip("fraud_txns:Q", title="Fraud transactions", format=",.0f"),
            alt.Tooltip("fraud_rate:Q", title="Fraud rate (%)", format=".2f"),
            alt.Tooltip("fraud_amount:Q", title="Fraud amount", format="$,.0f"),
        ],
    )
    labels = alt.Chart(g).mark_text(align="left", dx=4, fontSize=11).encode(
        x="fraud_rate:Q",
        y=alt.Y(f"{cat}:N", sort=alt.EncodingSortField(field="fraud_rate", order="descending")),
        text=alt.Text("fraud_rate:Q", format=".2f"),
    )
    return (bars + labels).properties(width="container", height=height)


def heat_chart(h: pd.DataFrame) -> alt.Chart:
    return alt.Chart(h).mark_rect().encode(
        x=alt.X("hour:O", title="Hour of day"),
        y=alt.Y("day:O", sort=L.DAY_ORDER, title=None),
        color=alt.Color("fraud_rate:Q", scale=alt.Scale(scheme="oranges"), title="Fraud rate (%)"),
        tooltip=[
            alt.Tooltip("day:N", title="Day"),
            alt.Tooltip("hour:O", title="Hour"),
            alt.Tooltip("transactions:Q", title="Transactions", format=",.0f"),
            alt.Tooltip("fraud_txns:Q", title="Fraud transactions", format=",.0f"),
            alt.Tooltip("fraud_rate:Q", title="Fraud rate (%)", format=".2f"),
        ],
    ).properties(width="container", height=250)


def donut(dist: pd.DataFrame) -> alt.Chart:
    return alt.Chart(dist).mark_arc(innerRadius=62, outerRadius=105).encode(
        theta=alt.Theta("transactions:Q"),
        color=alt.Color("tier:N", scale=TIER_SCALE, sort=L.TIER_ORDER, legend=alt.Legend(title=None, orient="bottom")),
        order=alt.Order("tier:N", sort="ascending"),
        tooltip=[
            alt.Tooltip("tier:N", title="Risk tier"),
            alt.Tooltip("transactions:Q", title="Transactions", format=",.0f"),
            alt.Tooltip("share:Q", title="Share (%)", format=".2f"),
            alt.Tooltip("confirmed_fraud:Q", title="Of which confirmed fraud", format=",.0f"),
        ],
    ).properties(height=270)


def line_xy(df_xy: pd.DataFrame, x: str, y: str, title: str, color: str, diagonal: bool = False, height: int = 280) -> alt.Chart:
    c = alt.Chart(df_xy).mark_line(color=color, strokeWidth=2.2, interpolate="step-after").encode(
        x=alt.X(f"{x}:Q", title=x.title(), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y(f"{y}:Q", title=y.title(), scale=alt.Scale(domain=[0, 1.02])),
        tooltip=[alt.Tooltip(f"{x}:Q", format=".3f"), alt.Tooltip(f"{y}:Q", format=".3f")],
    )
    if diagonal:
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="gray", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
        c = c + diag
    return c.properties(title=title, width="container", height=height)


def downsample(a: np.ndarray, b: np.ndarray, limit: int = 1500):
    if len(a) <= limit:
        return a, b
    idx = np.unique(np.linspace(0, len(a) - 1, limit).astype(int))
    return a[idx], b[idx]


def style_tier(df: pd.DataFrame, cols):
    styler = df.style
    for col in cols:
        if col in df.columns:
            styler = styler.map(lambda v: TIER_STYLE.get(str(v), ""), subset=[col])
    return styler


# =============================================================================
# PAGE 1: OVERVIEW
# =============================================================================
def page_overview():
    cur = L.kpis(view, has_label)
    old = L.kpis(prev, has_label) if len(prev) else None

    def delta_for(key, bad_when_up=True):
        if old is None or cur.get(key) is None or old.get(key) is None:
            return None, "off"
        d = L.pct_change(cur[key], old[key])
        return L.fmt_delta(d), ("inverse" if bad_when_up else "normal")

    r1 = st.columns(4)
    d, c = delta_for("total", False)
    with r1[0]:
        metric_card("Total transactions", L.fmt_int(cur["total"]), d, c)
    d, c = delta_for("value", False)
    with r1[1]:
        metric_card("Transaction value", L.fmt_money(cur["value"]), d, c)
    d, c = delta_for("fraud_txns")
    with r1[2]:
        metric_card("Fraud transactions", L.fmt_int(cur["fraud_txns"]), d, c, "Confirmed fraud (ground-truth label).")
    d, c = delta_for("fraud_amount")
    with r1[3]:
        metric_card("Fraud amount", L.fmt_money(cur["fraud_amount"]), d, c, "Total value of confirmed fraud transactions.")

    r2 = st.columns(4)
    d, c = delta_for("fraud_rate")
    with r2[0]:
        metric_card("Fraud rate", L.fmt_pct(cur["fraud_rate"], 3), d, c, "Fraud transactions / all transactions in the filtered view.")
    d, c = delta_for("high_risk")
    with r2[1]:
        metric_card("High-risk transactions", L.fmt_int(cur["high_risk"]), d, c, "Risk tier HIGH or CRITICAL.")
    with r2[2]:
        metric_card("Detection rate", L.fmt_pct(cur["recall"], 1), None, "off", "Recall: share of confirmed fraud the model flagged at the production threshold.")
    with r2[3]:
        metric_card("False positive rate", L.fmt_pct(cur["fpr"], 2), None, "off", "Legitimate transactions flagged / all legitimate transactions.")
    if old is None and has_time:
        note("Trend arrows need an earlier period of equal length inside the data; none is available for this selection.")

    st.markdown("#### Fraud activity trend")
    if not has_time:
        st.info("The data has no transaction_timestamp column, so time trends are unavailable.")
    else:
        t1, t2, _ = st.columns([1, 1, 2])
        gran = t1.radio("Granularity", ["Daily", "Weekly", "Monthly"], horizontal=True, key="ov_gran")
        metric = t2.radio("Measure", ["Transaction count", "Transaction amount"], horizontal=True, key="ov_metric")
        tr = L.trend(view, gran)
        if len(tr) < 2:
            st.info("Only one period in this selection. Widen the date range or choose a finer granularity.")
        else:
            show(trend_chart(tr, metric))
            note("Green (left axis): legitimate. Red solid: confirmed fraud. Orange dashed: flagged by the model (right axis).")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Transaction risk distribution")
        dist = L.risk_distribution(view)
        show(donut(dist))
        tbl = dist.rename(columns={"tier": "Risk tier", "transactions": "Transactions", "share": "Share (%)", "confirmed_fraud": "Confirmed fraud inside tier"})
        st.dataframe(tbl.style.format({"Share (%)": "{:.2f}", "Transactions": "{:,}", "Confirmed fraud inside tier": "{:,}"}), hide_index=True)
        note("Confirmed fraud is shown inside each tier, not as a separate slice, so no transaction is counted twice.")
    with right:
        st.markdown("#### Fraud by transaction type")
        g = L.group_fraud(view, "transaction_type")
        if g.empty or not has_label:
            st.info("Needs transaction_type and fraud_label columns.")
        else:
            show(rank_bars(g, "transaction_type"))
            with st.expander("Counts"):
                st.dataframe(g.rename(columns={"fraud_txns": "fraud", "fraud_amount": "fraud_amt"}), hide_index=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Highest-risk merchant categories")
        g = L.group_fraud(view, "merchant_category")
        if g.empty or not has_label:
            st.info("Needs merchant_category and fraud_label columns.")
        else:
            st.dataframe(
                g,
                hide_index=True,
                column_config={
                    "merchant_category": "Category",
                    "transactions": st.column_config.NumberColumn("Volume", format="%d"),
                    "fraud_txns": st.column_config.NumberColumn("Fraud txns", format="%d"),
                    "fraud_rate": st.column_config.ProgressColumn("Fraud rate (%)", format="%.2f", min_value=0.0, max_value=float(max(g["fraud_rate"].max(), 0.01))),
                    "fraud_amount": st.column_config.NumberColumn("Fraud amount", format="$%.0f"),
                    "volume": st.column_config.NumberColumn("Total value", format="$%.0f"),
                },
            )
    with right:
        st.markdown("#### Fraud hotspots")
        if "transaction_location" not in view.columns or not has_label:
            st.info("Needs transaction_location and fraud_label columns.")
        else:
            min_n = st.slider("Minimum transactions per location", 1, 200, 10, key="ov_minloc")
            g = L.group_fraud(view, "transaction_location", min_n=min_n).head(12)
            if g.empty:
                st.info("No location has that many transactions. Lower the minimum.")
            else:
                show(rank_bars(g, "transaction_location"))
                note("Ranked list rather than a map: the data holds location names, not coordinates.")

    st.markdown("#### Fraud activity by hour and day")
    h = L.hour_day_heat(view)
    if h.empty or not has_label:
        st.info("Needs transaction_timestamp and fraud_label columns.")
    else:
        show(heat_chart(h))
        note("Cells with few transactions can show extreme rates; hover for the counts behind each cell.")


# =============================================================================
# PAGE 2: FRAUD INTELLIGENCE
# =============================================================================
def page_intelligence():
    if not has_label:
        st.info("Fraud pattern analysis needs a fraud_label column.")
        return
    d = L.add_amount_band(view)
    dims = [c for c in ["transaction_type", "merchant_category", "payment_method", "device_type",
                        "authentication_method", "transaction_location", "amount_band"] if c in d.columns]
    pretty = {c: c.replace("_", " ").title() for c in dims}

    st.markdown("#### Fraud pattern matrix")
    if len(dims) < 2:
        st.info("Needs at least two of: transaction type, merchant category, payment method, device type, authentication method, location, amount.")
    else:
        c1, c2, c3 = st.columns(3)
        row = c1.selectbox("Rows", dims, index=0, format_func=pretty.get, key="pm_row")
        col = c2.selectbox("Columns", dims, index=1, format_func=pretty.get, key="pm_col")
        min_cell = c3.slider("Minimum transactions per cell", 1, 100, 10, key="pm_min")
        pm = L.pattern_matrix(d, row, col, min_cell)
        if row == col:
            st.info("Choose two different dimensions.")
        elif pm.empty:
            st.info("No cell reaches that many transactions. Lower the minimum.")
        else:
            chart = alt.Chart(pm).mark_rect().encode(
                x=alt.X(f"{col}:N", title=pretty[col]),
                y=alt.Y(f"{row}:N", title=pretty[row]),
                color=alt.Color("fraud_rate:Q", scale=alt.Scale(scheme="oranges"), title="Fraud rate (%)"),
                tooltip=[
                    alt.Tooltip(f"{row}:N", title=pretty[row]), alt.Tooltip(f"{col}:N", title=pretty[col]),
                    alt.Tooltip("transactions:Q", title="Transactions", format=",.0f"),
                    alt.Tooltip("fraud_txns:Q", title="Fraud transactions", format=",.0f"),
                    alt.Tooltip("fraud_rate:Q", title="Fraud rate (%)", format=".2f"),
                ],
            ).properties(width="container", height=max(200, 34 * pm[row].nunique()))
            show(chart)

    st.markdown("#### Fraud rate by single dimension")
    if dims:
        tabs = st.tabs([pretty[c] for c in dims])
        for tab, dim in zip(tabs, dims):
            with tab:
                g = L.group_fraud(d, dim)
                if g.empty:
                    st.info("No data.")
                else:
                    show(rank_bars(g.head(15), dim, height=max(160, 28 * min(len(g), 15))))

    st.markdown("#### Transaction amount analysis")
    if "transaction_amount" not in view.columns:
        st.info("Needs a transaction_amount column.")
    else:
        stats = L.amount_stats(view)
        left, right = st.columns([2, 1])
        with left:
            sample = view[["transaction_amount", "is_fraud"]]
            if len(sample) > 20000:
                sample = sample.sample(20000, random_state=0)
            dens = pd.DataFrame({
                "log_amount": np.log10(sample["transaction_amount"].clip(lower=0) + 1.0),
                "group": np.where(sample["is_fraud"] == 1, "Fraudulent", "Legitimate"),
            })
            if dens["group"].nunique() < 2:
                st.info("Only one class present in this selection, so there is nothing to compare.")
            else:
                chart = alt.Chart(dens).transform_density(
                    "log_amount", groupby=["group"], as_=["log_amount", "density"]
                ).mark_area(opacity=0.45).encode(
                    x=alt.X("log_amount:Q", title="Transaction amount",
                            axis=alt.Axis(labelExpr="'$' + format(pow(10, datum.value) - 1, ',.0f')")),
                    y=alt.Y("density:Q", title="Density"),
                    color=alt.Color("group:N", scale=alt.Scale(domain=["Legitimate", "Fraudulent"], range=[GREEN, RED]), legend=alt.Legend(title=None, orient="bottom")),
                ).properties(width="container", height=260)
                show(chart)
                note("Amounts are plotted on a log scale so small and very large transactions are both visible.")
        with right:
            if not stats.empty:
                st.dataframe(
                    stats.rename(columns={"group": "Group", "transactions": "Transactions", "average": "Average", "median": "Median"}).style.format(
                        {"Transactions": "{:,}", "Average": "${:,.2f}", "Median": "${:,.2f}"}),
                    hide_index=True)

    st.markdown("#### Behavioural risk signals")
    sig = L.behavioural_signals(view)
    if sig.empty:
        st.info("None of the signal columns (new_device, new_location, transaction_velocity, ...) are in the data.")
    else:
        left, right = st.columns([3, 2])
        with left:
            st.dataframe(
                sig, hide_index=True,
                column_config={
                    "signal": "Signal",
                    "affected_transactions": st.column_config.NumberColumn("Affected transactions", format="%d"),
                    "fraud_rate": st.column_config.ProgressColumn("Fraud rate (%)", format="%.2f", min_value=0.0, max_value=float(max(sig["fraud_rate"].max(), 0.01))),
                    "lift_vs_overall": st.column_config.NumberColumn("Lift vs overall", format="%.1fx", help="Fraud rate with the signal divided by the overall fraud rate."),
                    "fraud_contribution": st.column_config.NumberColumn("Fraud contribution (%)", format="%.1f", help="Share of all fraud that carries this signal."),
                    "rule": "Rule used",
                })
        with right:
            chart = alt.Chart(sig).mark_bar(color=RED).encode(
                x=alt.X("fraud_contribution:Q", title="Share of all fraud carrying the signal (%)"),
                y=alt.Y("signal:N", sort="-x", title=None),
                tooltip=[alt.Tooltip("signal:N"), alt.Tooltip("fraud_contribution:Q", format=".1f"), alt.Tooltip("fraud_rate:Q", format=".2f")],
            ).properties(width="container", height=max(180, 32 * len(sig)))
            show(chart)


# =============================================================================
# PAGE 3: TRANSACTION INVESTIGATION (+ alert queue)
# =============================================================================
def action_banner(action: str):
    a = str(action).upper()
    if any(k in a for k in ("BLOCK", "INVESTIGATE", "FREEZE")):
        st.error(f"Recommended action: {action}")
    elif any(k in a for k in ("REVIEW", "VERIFY", "STEP", "2FA", "AUTH")):
        st.warning(f"Recommended action: {action}")
    else:
        st.success(f"Recommended action: {action}")


def page_investigation():
    st.markdown("#### Transaction investigation")
    f1, f2, f3 = st.columns([2, 1, 1])
    query = f1.text_input("Search transaction or customer ID", "", key="inv_search")
    min_amt = f2.number_input("Minimum amount ($)", min_value=0.0, value=0.0, step=50.0, key="inv_minamt")
    max_rows_shown = f3.select_slider("Rows shown", options=[100, 250, 500, 1000], value=250, key="inv_rows")

    q = view
    if query:
        mask = pd.Series(False, index=q.index)
        for c in ("transaction_id", "customer_id"):
            if c in q.columns:
                mask |= q[c].astype(str).str.contains(query, case=False, na=False, regex=False)
        q = q[mask]
    if min_amt > 0 and "transaction_amount" in q.columns:
        q = q[q["transaction_amount"] >= min_amt]
    q = q.sort_values("risk_score", ascending=False).head(max_rows_shown).reset_index(drop=True)
    if q.empty:
        st.info("No transactions match the search.")
    else:
        merchant_col = "merchant_id" if "merchant_id" in q.columns else ("merchant_category" if "merchant_category" in q.columns else None)
        cols = [("transaction_id", "Transaction ID"), ("customer_id", "Customer ID"), ("ts", "Timestamp"),
                ("transaction_amount", "Amount"), ("transaction_type", "Type"), (merchant_col, "Merchant"),
                ("transaction_location", "Location"), ("device_type", "Device"),
                ("fraud_probability", "Fraud probability"), ("anomaly_score", "Anomaly score"),
                ("risk_score", "Risk score"), ("risk_tier", "Risk level"), ("fraud_status", "Fraud status"),
                ("recommended_action", "Recommended action")]
        cols = [(c, n) for c, n in cols if c and c in q.columns]
        table = q[[c for c, _ in cols]].rename(columns=dict(cols))
        fmt = {n: f for n, f in {"Amount": "${:,.2f}", "Fraud probability": "{:.1%}", "Anomaly score": "{:.3f}"}.items() if n in table.columns}
        styled = style_tier(table, ["Risk level"]).format(fmt)
        st.caption(f"{len(table):,} highest-risk matching transactions. Click a row's checkbox to open its investigation panel.")
        selected = None
        try:
            event = st.dataframe(styled, hide_index=True, on_select="rerun", selection_mode="single-row", key="inv_table")
            rows = event.selection.rows
            if rows:
                selected = q.iloc[rows[0]]
        except TypeError:
            st.dataframe(styled, hide_index=True)
        if "transaction_id" in q.columns:
            options = ["-"] + q["transaction_id"].astype(str).head(200).tolist()
            pick = st.selectbox("Or open by transaction ID", options, key="inv_pick")
            if pick != "-":
                selected = q[q["transaction_id"].astype(str) == pick].iloc[0]

        if selected is not None:
            st.markdown("---")
            st.markdown("##### Investigation panel")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Transaction information**")
                fields = [("transaction_id", "Transaction ID"), ("customer_id", "Customer ID"), ("ts", "Timestamp"),
                          ("transaction_amount", "Amount"), (merchant_col, "Merchant"), ("merchant_category", "Merchant category"),
                          ("transaction_type", "Transaction type"), ("payment_method", "Payment method"),
                          ("transaction_location", "Location"), ("device_type", "Device")]
                seen = set()
                for col, label in fields:
                    if col and col in selected.index and col not in seen and pd.notna(selected[col]):
                        seen.add(col)
                        v = selected[col]
                        v = f"${float(v):,.2f}" if col == "transaction_amount" else str(v)
                        st.markdown(f"{label}: **{v}**")
            with c2:
                st.markdown("**Model results**")
                st.metric("Fraud probability", f"{selected['fraud_probability'] * 100:.1f}%", help="Supervised model output.")
                if "anomaly_score" in selected.index:
                    st.metric("Anomaly score", f"{selected['anomaly_score']:.3f}", help="Unsupervised Isolation Forest output.")
                st.metric("Risk score", f"{int(selected['risk_score'])} / 100", help="Combined score used for tiers.")
                st.markdown(f"Risk level: **{selected['risk_tier']}**  \nFraud status: **{selected['fraud_status']}**")
            with c3:
                st.markdown("**Recommended action**")
                if "recommended_action" in selected.index:
                    action_banner(selected["recommended_action"])
                st.markdown("**Why was this transaction flagged?**")
                reasons = [r.strip() for r in str(selected.get("reasons", "")).split(";") if r.strip() and r.strip().lower() != "nan"]
                if reasons:
                    for r in reasons:
                        st.markdown(f"- {r}")
                else:
                    st.caption("The model recorded no explanation for this transaction.")

    # ---- Alert queue -------------------------------------------------------
    st.markdown("---")
    st.markdown("#### Alert queue")
    alerts = L.build_alerts(view)
    if alerts.empty:
        st.info("No HIGH or CRITICAL transactions in the current selection, so there are no alerts.")
        return
    status_map = st.session_state.setdefault("alert_status", {})
    alerts["status"] = alerts["alert_id"].map(lambda a: status_map.get(a, "New"))
    counts = alerts["status"].value_counts()
    k = st.columns(6)
    with k[0]:
        metric_card("New alerts", f"{counts.get('New', 0):,}")
    with k[1]:
        metric_card("High priority", f"{int((alerts['risk_level'] == 'CRITICAL').sum()):,}", help_text="Alerts at CRITICAL level.")
    with k[2]:
        metric_card("Under review", f"{counts.get('Under review', 0):,}")
    with k[3]:
        metric_card("Confirmed fraud", f"{counts.get('Confirmed fraud', 0):,}")
    with k[4]:
        metric_card("False positives", f"{counts.get('False positive', 0):,}")
    with k[5]:
        metric_card("Resolved", f"{counts.get('Resolved', 0):,}")

    a1, a2 = st.columns(2)
    level_f = a1.multiselect("Alert level", ["HIGH", "CRITICAL"], key="al_level", placeholder="All")
    status_f = a2.multiselect("Status", L.ALERT_STATUSES, key="al_status", placeholder="All")
    shown = alerts
    if level_f:
        shown = shown[shown["risk_level"].isin(level_f)]
    if status_f:
        shown = shown[shown["status"].isin(status_f)]
    shown = shown.head(200).reset_index(drop=True)
    edit_cols = ["alert_id", "transaction_id", "customer_id", "risk_score", "risk_level", "alert_reason", "created_at", "status", "recommended_action"]
    edited = st.data_editor(
        shown[edit_cols], hide_index=True, key="alert_editor",
        disabled=[c for c in edit_cols if c != "status"],
        column_config={
            "alert_id": "Alert ID", "transaction_id": "Transaction ID", "customer_id": "Customer ID",
            "risk_score": st.column_config.NumberColumn("Risk score", format="%d"),
            "risk_level": "Risk level", "alert_reason": "Alert reason",
            "created_at": st.column_config.DatetimeColumn("Created at", format="YYYY-MM-DD HH:mm"),
            "status": st.column_config.SelectboxColumn("Status", options=L.ALERT_STATUSES, required=True),
            "recommended_action": "Recommended action",
        })
    for aid, stt in zip(edited["alert_id"], edited["status"]):
        status_map[aid] = stt
    note("Alerts are generated from HIGH/CRITICAL transactions. Status changes are kept for this browser session only; the top 200 alerts by risk score are shown.")


# =============================================================================
# PAGE 4: CUSTOMER RISK
# =============================================================================
def page_customers():
    cust = L.customer_summary(view, cutoffs)
    if cust.empty:
        st.info("Needs a customer_id column.")
        return
    recent_cut = None
    recent = None
    if has_time:
        recent_cut = view["ts"].max() - pd.Timedelta(days=30)
        recent = view[(view["ts"] >= recent_cut) & view["risk_tier"].isin(["HIGH", "CRITICAL"])]["customer_id"].astype(str).nunique()

    k = st.columns(4)
    with k[0]:
        metric_card("Total customers", f"{len(cust):,}")
    with k[1]:
        metric_card("High-risk customers", f"{int(cust['risk_tier'].isin(['HIGH', 'CRITICAL']).sum()):,}", help_text="Customer risk score in the HIGH or CRITICAL tier.")
    with k[2]:
        metric_card("Customers with fraud history", f"{int((cust['fraud_count'] > 0).sum()):,}" if has_label else "n/a", help_text="At least one confirmed fraud transaction.")
    with k[3]:
        metric_card("Customers with recent alerts", f"{recent:,}" if recent is not None else "n/a", help_text="HIGH/CRITICAL transaction in the last 30 days of the selected period.")
    note("Customer risk score = the highest transaction risk score for that customer; tiers use the cut-offs set in the sidebar.")

    left, right = st.columns([2, 3])
    with left:
        st.markdown("#### Customer risk distribution")
        dist = cust["risk_tier"].value_counts().reindex(L.TIER_ORDER, fill_value=0).reset_index()
        dist.columns = ["tier", "customers"]
        chart = alt.Chart(dist).mark_bar().encode(
            x=alt.X("tier:N", sort=L.TIER_ORDER, title=None),
            y=alt.Y("customers:Q", title="Customers"),
            color=alt.Color("tier:N", scale=TIER_SCALE, legend=None),
            tooltip=[alt.Tooltip("tier:N", title="Tier"), alt.Tooltip("customers:Q", format=",.0f")],
        ).properties(width="container", height=260)
        show(chart)
    with right:
        st.markdown("#### Top high-risk customers")
        top = cust.head(25)
        table = top[["customer_id", "transactions", "total_amount", "fraud_count", "avg_transaction", "anomaly_score", "risk_score", "risk_tier"]].rename(
            columns={"customer_id": "Customer ID", "transactions": "Transactions", "total_amount": "Total amount",
                     "fraud_count": "Fraud count", "avg_transaction": "Average transaction", "anomaly_score": "Anomaly score",
                     "risk_score": "Risk score", "risk_tier": "Risk level"})
        st.dataframe(
            style_tier(table, ["Risk level"]).format({"Total amount": "${:,.2f}", "Average transaction": "${:,.2f}", "Anomaly score": "{:.3f}", "Risk score": "{:.0f}"}),
            hide_index=True)

    st.markdown("---")
    st.markdown("#### Customer investigation")
    choice = st.selectbox("Select a customer", cust["customer_id"].head(500).tolist(), key="cust_pick")
    cdf = view[view["customer_id"].astype(str) == str(choice)].sort_values("ts") if has_time else view[view["customer_id"].astype(str) == str(choice)]
    row = cust[cust["customer_id"] == str(choice)].iloc[0]
    m = st.columns(5)
    m[0].metric("Transactions", f"{int(row['transactions']):,}")
    m[1].metric("Total amount", L.fmt_money(row["total_amount"]))
    m[2].metric("Fraud count", f"{int(row['fraud_count']):,}" if has_label else "n/a")
    m[3].metric("Risk score", f"{row['risk_score']:.0f}")
    m[4].metric("Risk level", row["risk_tier"])

    if has_time and "transaction_amount" in cdf.columns and cdf["ts"].notna().any():
        pts = cdf.dropna(subset=["ts"])
        chart = alt.Chart(pts).mark_line(color=BLUE, opacity=0.5).encode(
            x=alt.X("ts:T", title=None), y=alt.Y("transaction_amount:Q", title="Amount ($)", axis=alt.Axis(format="$,.0f"))
        ) + alt.Chart(pts).mark_point(filled=True, size=70).encode(
            x="ts:T", y="transaction_amount:Q",
            color=alt.Color("risk_tier:N", scale=TIER_SCALE, legend=alt.Legend(title="Risk level", orient="bottom")),
            tooltip=[alt.Tooltip("transaction_id:N", title="Transaction") if "transaction_id" in pts.columns else alt.Tooltip("ts:T"),
                     alt.Tooltip("ts:T", title="Time"), alt.Tooltip("transaction_amount:Q", title="Amount", format="$,.2f"),
                     alt.Tooltip("risk_score:Q", title="Risk score", format=".0f"), alt.Tooltip("risk_tier:N", title="Risk level")],
        )
        st.markdown("**Transaction amount trend**")
        show(chart.properties(width="container", height=260))

    c1, c2, c3 = st.columns(3)
    for holder, col, title in ((c1, "transaction_location", "Locations"), (c2, "device_type", "Devices"), (c3, "merchant_category", "Merchant categories")):
        with holder:
            st.markdown(f"**{title}**")
            if col in cdf.columns:
                st.bar_chart(cdf[col].astype(str).value_counts())
            else:
                st.caption(f"No {col} column.")

    st.markdown("**Suspicious events**")
    flags = pd.Series(False, index=cdf.index)
    for col, test in (("new_device", lambda s: s == 1), ("new_location", lambda s: s == 1),
                      ("international_transaction", lambda s: s == 1), ("night_transaction", lambda s: s == 1),
                      ("failed_transaction_count", lambda s: s > 0)):
        if col in cdf.columns:
            flags |= test(cdf[col]).fillna(False)
    flags |= cdf["risk_tier"].isin(["HIGH", "CRITICAL"])
    ev_cols = [c for c in ["transaction_id", "ts", "transaction_amount", "transaction_type", "new_device", "new_location",
                           "international_transaction", "night_transaction", "failed_transaction_count", "risk_score", "risk_tier", "reasons"] if c in cdf.columns]
    events = cdf[flags][ev_cols]
    if events.empty:
        st.caption("No suspicious events for this customer in the selected period.")
    else:
        st.dataframe(events, hide_index=True)
    with st.expander("Full transaction history"):
        st.dataframe(cdf[[c for c in ["transaction_id", "ts", "transaction_amount", "transaction_type", "merchant_category", "transaction_location", "device_type", "risk_score", "risk_tier", "fraud_status"] if c in cdf.columns]], hide_index=True)


# =============================================================================
# PAGE 5: MODEL PERFORMANCE
# =============================================================================
def page_model():
    st.markdown("#### Fraud detection model performance")
    note("Fraud probability (supervised model), anomaly score (unsupervised Isolation Forest) and risk score (combined 0-100) are three different numbers. "
         "Metrics below describe the supervised fraud probability at the production threshold and follow the sidebar filters.")
    if not has_label:
        st.info("Model evaluation needs a fraud_label column in the data.")
    else:
        y = view["is_fraud"].to_numpy()
        p = view["fraud_probability"].to_numpy()
        roc = L.roc_curve_points(y, p)
        pr = L.pr_curve_points(y, p)
        cm = L.confusion_counts(view)
        mm = L.classification_metrics(cm)
        if roc is None or pr is None:
            st.warning("The current selection contains only one class, so ROC and PR curves cannot be drawn. Widen the filters.")
        else:
            cards = st.columns(7)
            vals = [("ROC-AUC", f"{roc[2]:.4f}"), ("PR-AUC", f"{pr[2]:.4f}"), ("Precision", L.fmt_pct(None if mm['precision'] is None else mm['precision'] * 100, 1)),
                    ("Recall", L.fmt_pct(None if mm['recall'] is None else mm['recall'] * 100, 1)), ("F1-score", "n/a" if mm["f1"] is None else f"{mm['f1']:.3f}"),
                    ("False positive rate", L.fmt_pct(None if mm['fpr'] is None else mm['fpr'] * 100, 2)), ("False negative rate", L.fmt_pct(None if mm['fnr'] is None else mm['fnr'] * 100, 2))]
            for holder, (label, value) in zip(cards, vals):
                with holder:
                    metric_card(label, value)
            note(f"Production threshold {threshold:.2f} ({threshold_source}). PR-AUC is reported as average precision.")

            c1, c2 = st.columns(2)
            with c1:
                fx, fy = downsample(roc[0], roc[1])
                show(line_xy(pd.DataFrame({"false positive rate": fx, "true positive rate": fy}), "false positive rate", "true positive rate",
                             f"ROC curve (AUC {roc[2]:.4f})", BLUE, diagonal=True))
            with c2:
                rx, ry = downsample(pr[0], pr[1])
                base_rate = float(y.mean())
                chart = line_xy(pd.DataFrame({"recall": rx, "precision": ry}), "recall", "precision", f"Precision-Recall curve (AP {pr[2]:.4f})", RED)
                base_line = alt.Chart(pd.DataFrame({"y": [base_rate]})).mark_rule(color="gray", strokeDash=[4, 4]).encode(y="y:Q")
                show((chart + base_line).properties(width="container", height=280))
                note(f"Dashed line: precision of a random model (the fraud rate, {base_rate * 100:.2f}%). With rare fraud, PR-AUC is the more honest summary than ROC-AUC.")

            st.markdown("#### Confusion matrix")
            cells = pd.DataFrame([
                {"actual": "Actual: legitimate", "predicted": "Predicted: not flagged", "kind": "True negative", "count": cm["tn"]},
                {"actual": "Actual: legitimate", "predicted": "Predicted: flagged", "kind": "False positive", "count": cm["fp"]},
                {"actual": "Actual: fraud", "predicted": "Predicted: not flagged", "kind": "False negative (missed fraud)", "count": cm["fn"]},
                {"actual": "Actual: fraud", "predicted": "Predicted: flagged", "kind": "True positive", "count": cm["tp"]},
            ])
            domain = ["True negative", "False positive", "False negative (missed fraud)", "True positive"]
            base_cm = alt.Chart(cells).encode(
                x=alt.X("predicted:N", title=None, sort=["Predicted: not flagged", "Predicted: flagged"]),
                y=alt.Y("actual:N", title=None, sort=["Actual: legitimate", "Actual: fraud"]),
            )
            rect = base_cm.mark_rect().encode(
                color=alt.Color("kind:N", scale=alt.Scale(domain=domain, range=["#BFE3D0", "#F6D99A", DARKRED, GREEN]), legend=alt.Legend(title=None, orient="bottom")),
                tooltip=[alt.Tooltip("kind:N", title="Outcome"), alt.Tooltip("count:Q", title="Transactions", format=",.0f")],
            )
            text = base_cm.mark_text(fontSize=20, fontWeight="bold").encode(text=alt.Text("count:Q", format=",.0f"),
                                                                           color=alt.condition(alt.datum.kind == "False negative (missed fraud)", alt.value("white"), alt.value("#1F2937")))
            cm_col, txt_col = st.columns([2, 1])
            with cm_col:
                show((rect + text).properties(width="container", height=230))
            with txt_col:
                st.metric("Missed fraud (false negatives)", f"{cm['fn']:,}", help="Fraud the model did not flag. These are direct financial losses.")
                missed_amt = view.loc[(~view["flagged"]) & (view["is_fraud"] == 1), "transaction_amount"].sum() if "transaction_amount" in view.columns else None
                st.metric("Value of missed fraud", L.fmt_money(missed_amt))
                st.metric("False alarms (false positives)", f"{cm['fp']:,}")

    # ---- model comparison --------------------------------------------------
    st.markdown("#### Model comparison")
    path = Path(MODEL_COMPARISON_FILE)
    if path.exists():
        comp = read_csv_file(str(path), os.path.getmtime(path))
        comp = comp.copy()
        if "recall" in comp.columns and "false_negative_rate" not in comp.columns:
            comp["false_negative_rate"] = 1 - pd.to_numeric(comp["recall"], errors="coerce")
        pref = ["model", "roc_auc", "pr_auc", "precision", "recall", "f1", "false_negative_rate", "false_positives", "false_negatives", "estimated_business_cost_usd"]
        cols = [c for c in pref if c in comp.columns] or comp.columns.tolist()
        shown = comp[cols]
        numeric = [c for c in cols if c != "model" and pd.api.types.is_numeric_dtype(shown[c])]
        styler = shown.style.format({c: ("{:,.0f}" if c in ("false_positives", "false_negatives", "estimated_business_cost_usd") else "{:.4f}") for c in numeric})
        if "pr_auc" in shown.columns:
            styler = styler.highlight_max(subset=["pr_auc"], color="#CFE8D8")
        st.dataframe(styler, hide_index=True)
        note("Values come from your model comparison file (held-out test split). The highest PR-AUC is highlighted. "
             "Curves above are drawn for the production model only, because per-model scores are not stored.")
    else:
        st.info("Model comparison file not found. Run training to create it.")

    # ---- feature importance ------------------------------------------------
    st.markdown("#### Feature importance")
    fpath = Path(FEATURE_IMPORTANCE_FILE)
    if fpath.exists():
        fi = read_csv_file(str(fpath), os.path.getmtime(fpath))
        if {"feature", "importance"} <= set(fi.columns):
            topn = st.slider("Features shown", 5, 20, 15, key="fi_n")
            top = fi.sort_values("importance", ascending=False).head(topn)
            chart = alt.Chart(top).mark_bar(color=BLUE).encode(
                x=alt.X("importance:Q", title="Importance"),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("importance:Q", format=".4f")],
            ).properties(width="container", height=max(200, 24 * len(top)))
            show(chart)
            note("Only features recorded by the trained model are listed.")
        else:
            st.info("Feature importance file needs 'feature' and 'importance' columns.")
    else:
        st.info("Feature importance file not found. Run training to create it.")

    # ---- threshold analysis ------------------------------------------------
    st.markdown("#### Threshold analysis")
    if not has_label:
        st.info("Needs a fraud_label column.")
    else:
        sel = st.slider("Fraud probability threshold", 0.05, 0.95, float(min(max(threshold, 0.05), 0.95)), 0.05, key="thr_sel")
        y = view["is_fraud"].to_numpy()
        p = view["fraud_probability"].to_numpy()
        grid = np.round(np.arange(0.05, 0.951, 0.05), 2)
        sweep = L.threshold_sweep(y, p, np.unique(np.r_[grid, sel, threshold]))
        at = L.threshold_sweep(y, p, [sel]).iloc[0]
        m = st.columns(6)
        m[0].metric("Precision", L.fmt_pct(None if pd.isna(at["precision"]) else at["precision"] * 100, 1))
        m[1].metric("Recall", L.fmt_pct(None if pd.isna(at["recall"]) else at["recall"] * 100, 1))
        m[2].metric("F1", "n/a" if pd.isna(at["f1"]) else f"{at['f1']:.3f}")
        m[3].metric("False positives", f"{int(at['false_positives']):,}")
        m[4].metric("False negatives", f"{int(at['false_negatives']):,}")
        m[5].metric("Flagged transactions", f"{int(at['flagged']):,}")
        long = sweep.melt(id_vars="threshold", value_vars=["precision", "recall", "f1"], var_name="metric", value_name="value").dropna()
        lines = alt.Chart(long).mark_line(point=True).encode(
            x=alt.X("threshold:Q", title="Threshold", scale=alt.Scale(domain=[0.05, 0.95])),
            y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1.02])),
            color=alt.Color("metric:N", scale=alt.Scale(domain=["precision", "recall", "f1"], range=[BLUE, RED, GREEN]), legend=alt.Legend(title=None, orient="bottom")),
            tooltip=[alt.Tooltip("metric:N"), alt.Tooltip("threshold:Q", format=".2f"), alt.Tooltip("value:Q", format=".3f")],
        )
        prod = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(color="gray", strokeDash=[6, 4]).encode(x="t:Q")
        chosen = alt.Chart(pd.DataFrame({"t": [sel]})).mark_rule(color=ORANGE, strokeWidth=2).encode(x="t:Q")
        show((lines + prod + chosen).properties(width="container", height=280))
        note(f"Grey dashed line: production threshold ({threshold:.2f}, {threshold_source}). Orange line: the threshold you are exploring.")
        counts = sweep.melt(id_vars="threshold", value_vars=["false_positives", "false_negatives"], var_name="error", value_name="count")
        counts["error"] = counts["error"].map({"false_positives": "False positives", "false_negatives": "False negatives"})
        show(alt.Chart(counts).mark_line(point=True).encode(
            x=alt.X("threshold:Q", title="Threshold"), y=alt.Y("count:Q", title="Transactions"),
            color=alt.Color("error:N", scale=alt.Scale(domain=["False positives", "False negatives"], range=[AMBER, DARKRED]), legend=alt.Legend(title=None, orient="bottom")),
            tooltip=[alt.Tooltip("error:N"), alt.Tooltip("threshold:Q", format=".2f"), alt.Tooltip("count:Q", format=",.0f")],
        ).properties(width="container", height=220))

    # ---- risk score framework ---------------------------------------------
    st.markdown("#### Risk score framework")
    edges = [0, cutoffs["MEDIUM"], cutoffs["HIGH"], cutoffs["CRITICAL"], 100]
    scale_df = pd.DataFrame({
        "tier": L.TIER_ORDER, "start": edges[:-1], "end": [e - 1 for e in edges[1:-1]] + [100],
    })
    scale_df["label"] = scale_df["tier"] + " (" + scale_df["start"].astype(str) + "-" + scale_df["end"].astype(str) + ")"
    test = st.slider("Try a risk score", 0, 100, int(view["risk_score"].median()), key="rs_test")
    seg = alt.Chart(scale_df).mark_bar(size=34).encode(
        x=alt.X("start:Q", title="Risk score", scale=alt.Scale(domain=[0, 101])), x2="end:Q",
        color=alt.Color("tier:N", scale=TIER_SCALE, legend=None),
        tooltip=[alt.Tooltip("label:N", title="Tier")])
    names = alt.Chart(scale_df).mark_text(dy=-26, fontSize=11).encode(x="start:Q", text="label:N")
    marker = alt.Chart(pd.DataFrame({"x": [test]})).mark_rule(color="white", strokeWidth=3).encode(x="x:Q")
    marker_edge = alt.Chart(pd.DataFrame({"x": [test]})).mark_rule(color="#111827", strokeWidth=1).encode(x="x:Q")
    show((seg + names + marker_edge + marker).properties(width="container", height=90))
    tier_now = L.assign_tier(pd.Series([test]), cutoffs).iloc[0]
    st.caption(f"A risk score of {test} falls in the {tier_now} tier. Change the cut-offs in the sidebar under 'Risk score tiers'; every page updates.")


# =============================================================================
# Router
# =============================================================================
{
    "Overview": page_overview,
    "Fraud Intelligence": page_intelligence,
    "Transaction Investigation": page_investigation,
    "Customer Risk": page_customers,
    "Model Performance": page_model,
}[page]()

st.divider()
st.caption("Fraud Risk Intelligence - predictions are model estimates that support, not replace, analyst judgement.")