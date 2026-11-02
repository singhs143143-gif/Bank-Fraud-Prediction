# 📊 Dashboard & Enterprise Analytics Engine Guide

> **Intelligent Banking Transaction & Fraud Detection System**  
> Complete documentation for the **Streamlit Web Console (`app.py`)**, **Calculation Engine (`dashboard_logic.py`)**, and **Power BI Enterprise Star Schema (`powerbi_dashboard_requirements.md`)**.

---

## 1. Overview & Architecture

The fraud analytics suite provides dual reporting and operational triage interfaces:

1. **Streamlit Real-Time Fraud Intelligence Console (`app.py`)**:
   - Low-latency interactive web application built with Streamlit, Altair, and Python.
   - Powered by real-time dual-engine inference (`XGBoost 2.0+` + `Isolation Forest`).
   - Serves frontline fraud analysts, risk operations teams, and compliance officers.

2. **Power BI Executive Star Schema (`powerbi_dashboard_requirements.md`)**:
   - Enterprise BI reporting model designed for Chief Risk Officers (CRO) and executive management.
   - DirectLake / Import mode Star Schema with 25+ DAX measures and 4 dedicated reporting pages.

3. **Pure Pandas/NumPy Calculation Layer (`dashboard_logic.py`)**:
   - Decoupled mathematical logic ensuring zero presentation logic is mixed with calculations.
   - Guaranteed deterministic metric aggregation, percentile ranking, and time-series bucketing.

---

## 2. Streamlit Web Console: The 6 Operational Workstations

```text
========================================================================================
🛡️ NOVA SENTINEL • FRAUD INTELLIGENCE CONSOLE
Status: ● LIVE INFERENCE ACTIVE | Engine: Dual ML Pipeline (XGBoost + Isolation Forest)
========================================================================================
[ 📊 Executive Overview ] [ 🧠 Fraud Intelligence ] [ 🔍 Case Investigation ]
[ ⚡ Live Scoring Sandbox ] [ 👥 Customer Risk ] [ 📈 Model Diagnostics ]
========================================================================================
```

### Module 1: 📊 Executive Overview
* **Financial Exposure Metrics**: Real-time KPI tiles displaying Gross Volume, Processed Capital, Confirmed Fraud Losses, and Losses Prevented.
* **Temporal Trend Analysis**: Dual-axis line and area charts showing legitimate operational volume (left axis) against fraud incident trajectory (right axis) across Daily, Weekly, and Monthly intervals.
* **Population Donut Chart**: Proportional breakdown of transaction population across the four calibrated operational tiers: `LOW`, `MEDIUM`, `HIGH`, and `CRITICAL`.
* **Channel & Merchant Breakdown**: Vulnerability analysis across payment rails (`TRANSFER`, `CASH_OUT`, etc.) and sectors (`Luxury Goods`, `Electronics`, etc.).

### Module 2: 🧠 Fraud Intelligence
* **Multidimensional Cross-Correlation Matrix**: Intersect any two dimensions (e.g., Payment Method vs Authentication Method) to identify high-density attack vectors.
* **Behavioral Signal Lift Ranking**: Measures empirical risk lift multipliers for signatures such as `new_device`, `failed_transaction_count`, `unusual_auth_flag`, and `velocity_burst`.
* **Ticket Size Density**: Log-scale probability density curves comparing legitimate vs fraudulent spend distributions.

### Module 3: 🔍 Case Investigation & Alert Desk
* **Search & Slicers**: Filter transactions in real time by Transaction ID, Customer ID, minimum dollar value, or operational risk tier.
* **Inspection Dossier**: Side-by-side view of transaction attributes, hardware device fingerprints, and location data.
* **Deterministic AI Root-Cause Chips**: Explanatory tags detailing exactly why a transaction was flagged (e.g., *Balance Drain 92%*, *Cross-Border International Rail*, *Unrecognized Device*).
* **Operational Action Triggers**: Instant one-click action buttons:
  - `[🚫 Block Card]` — Freezes customer card token.
  - `[⚠️ Step-Up 2FA]` — Dispatches automated out-of-band biometric challenge.
  - `[✅ Whitelist]` — Marks transaction clean and dismisses alert.

### Module 4: ⚡ Live Scoring Sandbox (Real-Time Simulator)
* **Real-Time Single-Transaction Inference**: Test any arbitrary transaction payload through the production XGBoost and Isolation Forest pipelines in sub-5ms.
* **One-Click Attack Presets**:
  - `🚨 Scenario A: Account Takeover & Rapid Drain` ($9,450 wire transfer, new device, 5,500km away, 2 failed OTPs, overnight).
  - `🚨 Scenario B: SIM Swap & Nighttime Carding` ($4,200 cash out, electronics merchant, new device, international).
  - `🟢 Scenario C: Legitimate Coffee & Groceries` ($18.50 POS payment, trusted device, local).
  - `🟡 Scenario D: High-Value Domestic Salary Transfer` ($6,500 net banking, domestic, verified account).
* **Instant Decision Output**: Large radial composite risk score (0–100), decision badge, probability %, and root-cause explanations.

### Module 5: 👥 Customer Risk Profiles
* **Customer Entity Profiling**: Aggregate risk index computed at the account holder level.
* **Top 20 Watchlist Table**: Pinpoints accounts with repeat fraud history, velocity anomalies, and high balance depletion rates.

### Module 6: 📈 Model Diagnostics & Threshold Economics
* **Model Benchmark Curves**: ROC curve (`AUC = 0.9999`) and Precision-Recall curve (`PR-AUC = 0.9978`).
* **Confusion Matrix**: Instant breakdown of True Positives, False Positives, False Negatives, and True Negatives.
* **Interactive Economic Cost Calculator**: Balances asymmetric financial business costs:
  $$\text{Total Business Exposure} = (\text{FN} \times \$1,500) + (\text{FP} \times \$35)$$
* **Feature Importance Ranking**: Top 15 XGBoost Gain features displayed dynamically.

---

## 3. Calculation Engine: `dashboard_logic.py`

Located at both `dashboard/dashboard_logic.py` and `src/dashboard_logic.py`, this module isolates core business calculations:

| Function | Output & Purpose |
| :--- | :--- |
| `kpis(df, has_label)` | Computes total transactions, gross value, confirmed fraud count, fraud amount, fraud rate %, high risk count, model recall, and false positive rate. |
| `trend(df, gran)` | Groups transactions chronologically by Daily (`D`), Weekly (`W`), or Monthly (`MS`) intervals. |
| `risk_distribution(df)` | Aggregates volume, percentage share, and confirmed fraud within each operational tier. |
| `group_fraud(df, col, min_n)` | Computes volume, fraud count, fraud rate %, and total fraud exposure grouped by any categorical dimension. |
| `pattern_matrix(df, r, c, min_n)`| Builds a cross-tabulated correlation matrix with fraud rates across two arbitrary feature columns. |
| `behavioural_signals(df)` | Measures empirical fraud rate and risk lift multiplier for behavioral attack flags (`new_device`, `failed_logins`, etc.). |
| `threshold_sweep(y, p, grid)` | Calculates Precision, Recall, F1, False Positives, and False Negatives across an array of probability thresholds. |
| `customer_summary(df, cutoffs)` | Aggregates transactions, total spend, fraud occurrences, and average risk score at the `customer_id` level. |

---

## 4. Power BI Star Schema & DAX Reference

For organizations deploying enterprise business intelligence via Power BI Desktop, Power BI Report Server, or Microsoft Fabric:

### Star Schema Entity-Relationship Model

```text
                  ┌──────────────┐
                  │   DIM_DATE   │
                  └──────┬───────┘
                         │ 1:N
┌──────────────┐  ┌──────┴───────────┐  ┌──────────────┐
│ DIM_CUSTOMER ├──┤ FACT_TRANSACTIONS ├──┤ DIM_MERCHANT │
└──────────────┘  └──────┬───────────┘  └──────────────┘
                    1:1  │  1:N
                  ┌──────┴───────────┐
                  │ FACT_PREDICTIONS │
                  └──────┬───────────┘
                    1:N  │
                  ┌──────┴───────────┐
                  │   FACT_ALERTS    │
                  └──────────────────┘
```

### Core DAX Measures (Sample)

```dax
-- Operational Fraud Recall / Detection Rate
Fraud Detection Rate (%) = 
DIVIDE(
    CALCULATE(COUNTROWS(fact_transactions), fact_transactions[fraud_label] = 1, RELATED(fact_predictions[risk_level]) = "HIGH"),
    CALCULATE(COUNTROWS(fact_transactions), fact_transactions[fraud_label] = 1),
    0
) * 100

-- Financial Value of Prevented Fraud
Fraud Losses Prevented ($) = 
CALCULATE(
    SUM(fact_transactions[transaction_amount]),
    fact_transactions[fraud_label] = 1,
    RELATED(fact_predictions[risk_level]) = "HIGH"
)

-- Net Business Exposure Loss
Net Business Loss ($) = 
([False Negatives (FN)] * 1500) + ([False Positives (FP)] * 35)
```

See [powerbi_dashboard_requirements.md](powerbi_dashboard_requirements.md) for the complete 25+ DAX library, visual layout specifications, and conditional formatting rules.

---

## 5. Quick Launch Instructions

### 1. Launch Streamlit Web Console
```powershell
# Windows Batch Launcher:
run_dashboard.bat

# Or via Command Line:
.\.venv\Scripts\activate
streamlit run app.py
```
Open **`http://localhost:8501`** in any web browser.

### 2. Connect Power BI Desktop
1. Open Power BI Desktop.
2. Select **Get Data -> SQL Server / PostgreSQL / ODBC**.
3. Connect to your database initialized with `sql/schema.sql` and `sql/transaction_analytics.sql`.
4. Import `reports/model_comparison.csv` and `reports/feature_importance.csv`.
5. Copy DAX definitions from `powerbi_dashboard_requirements.md` into your `_Measures` table.
