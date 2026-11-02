"""
create_notebooks.py
-------------------
Generates the 6 production-grade Jupyter notebooks requested in Section 2.
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(source):
    lines = [s + "\n" for s in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code_cell(source):
    lines = [s + "\n" for s in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

# ==============================================================================
# NOTEBOOK 1: 01_data_exploration.ipynb
# ==============================================================================
nb1_cells = [
    md_cell("# 📊 Notebook 01: Exploratory Data Analysis (EDA)\n## Intelligent Banking Transaction & Fraud Detection System\n\nThis notebook conducts in-depth exploratory analysis on the raw 300,000 transaction dataset, analyzing distributions, class imbalance, diurnal patterns, and fraud risk indicators."),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path.cwd().parent
sys.path.insert(0, str(ROOT_DIR))
from src.config import RAW_DATA_FILE

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)"""),
    md_cell("### 1. Load Raw Dataset"),
    code_cell("""df = pd.read_csv(RAW_DATA_FILE, low_memory=False)
print(f"Dataset Dimensions: {df.shape[0]:,} rows x {df.shape[1]} columns")
df.head()"""),
    md_cell("### 2. Class Imbalance Analysis\nFraudulent transactions are an extreme minority class in banking systems."),
    code_cell("""fraud_counts = df["fraud_label"].value_counts()
fraud_rate = df["fraud_label"].mean() * 100
print(f"Legitimate Transactions (0): {fraud_counts[0]:,} ({100 - fraud_rate:.2f}%)")
print(f"Fraudulent Transactions (1): {fraud_counts[1]:,} ({fraud_rate:.2f}%)")

plt.figure(figsize=(6, 4))
sns.countplot(x="fraud_label", data=df, palette=["#059669", "#DC2626"])
plt.title("Severe Class Imbalance Distribution (2.1% Fraud)")
plt.xlabel("Fraud Status (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.show()"""),
    md_cell("### 3. Transaction Amount Distribution (Log-Scale)"),
    code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

sns.histplot(df["transaction_amount"], bins=50, kde=True, ax=ax1, color="#2563EB")
ax1.set_title("Raw Transaction Amounts ($)")

sns.histplot(np.log1p(df["transaction_amount"]), bins=50, kde=True, ax=ax2, color="#7C3AED")
ax2.set_title("Log1p Transformed Transaction Amounts")
plt.show()"""),
    md_cell("### 4. Fraud Rate by Merchant Category"),
    code_cell("""merch_summary = df.groupby("merchant_category")["fraud_label"].agg(["count", "sum", "mean"]).reset_index()
merch_summary["mean"] = merch_summary["mean"] * 100
merch_summary = merch_summary.sort_values(by="mean", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x="mean", y="merchant_category", data=merch_summary, palette="Reds_r")
plt.title("Fraud Incidence Rate (%) by Merchant Category")
plt.xlabel("Fraud Rate (%)")
plt.ylabel("Merchant Category")
plt.show()"""),
    md_cell("### 5. Diurnal Patterns: Fraud by Hour of the Day"),
    code_cell("""hourly = df.groupby("transaction_hour")["fraud_label"].mean() * 100
plt.figure(figsize=(10, 4))
hourly.plot(kind="bar", color="#DC2626")
plt.title("Fraud Probability (%) by Hour of Day (0 to 23)")
plt.xlabel("Hour of Day")
plt.ylabel("Fraud Rate (%)")
plt.show()""")
]

# ==============================================================================
# NOTEBOOK 2: 02_data_cleaning.ipynb
# ==============================================================================
nb2_cells = [
    md_cell("# 🧹 Notebook 02: Data Quality & Cleaning Pipeline\n## Data Validation, Deduplication, and Chronological Splitting\n\nThis notebook demonstrates rigorous data quality audits, missing value imputation, outlier handling, and time-aware dataset splitting."),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path.cwd().parent
sys.path.insert(0, str(ROOT_DIR))
from src.config import RAW_DATA_FILE
from src.data_processing import audit_and_clean_data, split_chronological"""),
    md_cell("### 1. Execute Data Quality Audits"),
    code_cell("""df_raw = pd.read_csv(RAW_DATA_FILE, low_memory=False)
df_clean = audit_and_clean_data(df_raw)
print(f"Cleaned DataFrame Shape: {df_clean.shape}")"""),
    md_cell("### 2. Chronological Splitting (70% Train, 15% Validation, 15% Test)\nTransactions are strictly ordered chronologically to prevent temporal data leakage."),
    code_cell("""train, val, test = split_chronological(df_clean)
print(f"Train Partition : {len(train):,} records | Date range: {train['transaction_timestamp'].min()} to {train['transaction_timestamp'].max()}")
print(f"Val Partition   : {len(val):,} records | Date range: {val['transaction_timestamp'].min()} to {val['transaction_timestamp'].max()}")
print(f"Test Partition  : {len(test):,} records | Date range: {test['transaction_timestamp'].min()} to {test['transaction_timestamp'].max()}")"""),
    md_cell("### 3. Verify Split Imbalance Consistency"),
    code_cell("""summary = pd.DataFrame({
    "Partition": ["Train", "Validation", "Test"],
    "Records": [len(train), len(val), len(test)],
    "Fraud Rate (%)": [train["fraud_label"].mean() * 100, val["fraud_label"].mean() * 100, test["fraud_label"].mean() * 100]
})
summary""")
]

# ==============================================================================
# NOTEBOOK 3: 03_feature_engineering.ipynb
# ==============================================================================
nb3_cells = [
    md_cell("# ⚙️ Notebook 03: Feature Engineering\n## Ratios, Velocity Signals, Cyclic Time & Behavioral Proxies\n\nConstructs domain-specific features without target leakage."),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path.cwd().parent
sys.path.insert(0, str(ROOT_DIR))
from src.config import TRAIN_FILE
from src.feature_engineering import engineer_features, get_feature_columns"""),
    md_cell("### 1. Load Training Data & Engineer Features"),
    code_cell("""train_raw = pd.read_csv(TRAIN_FILE, low_memory=False)
train_fe = engineer_features(train_raw)
print(f"Features Generated. New Column Count: {train_fe.shape[1]}")
train_fe[["transaction_id", "log_transaction_amount", "amount_to_balance_ratio", "velocity_surge_ratio", "behavioral_risk_proxy"]].head()"""),
    md_cell("### 2. Behavioral Risk Proxy vs Actual Fraud"),
    code_cell("""plt.figure(figsize=(8, 4))
sns.kdeplot(data=train_fe, x="behavioral_risk_proxy", hue="fraud_label", common_norm=False, palette=["#059669", "#DC2626"])
plt.title("Behavioral Risk Proxy Score Distribution by Class")
plt.xlabel("Heuristic Risk Proxy (0 to 1)")
plt.show()""")
]

# ==============================================================================
# NOTEBOOK 4: 04_fraud_modeling.ipynb
# ==============================================================================
nb4_cells = [
    md_cell("# 🤖 Notebook 04: Machine Learning Fraud Modeling\n## Preprocessing, SMOTE, Logistic Regression, Random Forest & XGBoost with 5-Fold CV\n\nTrains and tunes supervised classifiers with proper handling of class imbalance."),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path.cwd().parent
sys.path.insert(0, str(ROOT_DIR))
from src.config import REPORTS_DIR"""),
    md_cell("### 1. Review 5-Fold Stratified Cross-Validation Results"),
    code_cell("""cv_path = REPORTS_DIR / "cross_validation_results.csv"
if cv_path.exists():
    cv_df = pd.read_csv(cv_path)
    print("5-Fold Cross-Validation Performance on Training Data:")
    display(cv_df)
else:
    print("CV report file not found. Run train_models.py to generate.")""")
]

# ==============================================================================
# NOTEBOOK 5: 05_anomaly_detection.ipynb
# ==============================================================================
nb5_cells = [
    md_cell("# 🔍 Notebook 05: Unsupervised Anomaly Detection\n## Isolation Forest for Zero-Day Fraud & Novelty Detection\n\nEvaluates Isolation Forest trained without labels to uncover novel attack patterns."),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path.cwd().parent
sys.path.insert(0, str(ROOT_DIR))
from src.config import ISO_MODEL_FILE, TEST_FILE
from src.anomaly_detection import load_isolation_forest, get_anomaly_scores
from src.feature_engineering import engineer_features
from src.preprocessing import load_pipeline, transform"""),
    md_cell("### 1. Load Test Data & Generate Anomaly Scores"),
    code_cell("""test_df = pd.read_csv(TEST_FILE, low_memory=False)
test_fe = engineer_features(test_df)
pipeline = load_pipeline()
X_test, y_test = transform(pipeline, test_fe)

iso = load_isolation_forest()
anomaly_scores = get_anomaly_scores(iso, X_test)

test_df["anomaly_score"] = anomaly_scores
test_df.groupby("fraud_label")["anomaly_score"].describe()"""),
    md_cell("### 2. Anomaly Score Distribution"),
    code_cell("""plt.figure(figsize=(8, 4))
sns.histplot(data=test_df, x="anomaly_score", hue="fraud_label", bins=40, kde=True, palette=["#059669", "#DC2626"])
plt.title("Isolation Forest Anomaly Score Distribution (0 = Normal, 1 = Outlier)")
plt.xlabel("Normalized Anomaly Score")
plt.show()""")
]

# ==============================================================================
# NOTEBOOK 6: 06_model_evaluation.ipynb
# ==============================================================================
nb6_cells = [
    md_cell("# 📈 Notebook 06: Comprehensive Model Evaluation\n## Benchmarking, Threshold Optimization & Feature Importance\n\nEvaluates all supervised and unsupervised models on the held-out test set."),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path.cwd().parent
sys.path.insert(0, str(ROOT_DIR))
from src.config import MODEL_COMPARISON_FILE, FEATURE_IMPORTANCE_FILE, REPORTS_DIR"""),
    md_cell("### 1. Final Model Comparison on Held-Out Test Set"),
    code_cell("""comp_df = pd.read_csv(MODEL_COMPARISON_FILE)
comp_df"""),
    md_cell("### 2. Threshold Sensitivity Analysis"),
    code_cell("""thresh_path = REPORTS_DIR / "threshold_analysis.csv"
if thresh_path.exists():
    th_df = pd.read_csv(thresh_path)
    plt.figure(figsize=(9, 4.5))
    plt.plot(th_df["threshold"], th_df["precision"], marker="o", label="Precision", color="#2563EB")
    plt.plot(th_df["threshold"], th_df["recall"], marker="s", label="Recall", color="#DC2626")
    plt.plot(th_df["threshold"], th_df["f1"], marker="^", label="F1-Score", color="#059669")
    plt.axvline(x=0.40, color="gray", linestyle="--", label="Optimal Threshold (0.40)")
    plt.title("Precision, Recall, and F1-Score Across Probability Thresholds")
    plt.xlabel("Probability Classification Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.show()"""),
    md_cell("### 3. XGBoost Feature Importance Ranking"),
    code_cell("""fi_df = pd.read_csv(FEATURE_IMPORTANCE_FILE)
plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=fi_df.head(15), palette="Blues_r")
plt.title("Top 15 Most Important Features in XGBoost Fraud Detection")
plt.xlabel("Relative Feature Importance (Gain)")
plt.show()""")
]

notebooks = {
    "01_data_exploration.ipynb": nb1_cells,
    "02_data_cleaning.ipynb": nb2_cells,
    "03_feature_engineering.ipynb": nb3_cells,
    "04_fraud_modeling.ipynb": nb4_cells,
    "05_anomaly_detection.ipynb": nb5_cells,
    "06_model_evaluation.ipynb": nb6_cells,
}

for name, cells in notebooks.items():
    nb_content = make_notebook(cells)
    out_file = NOTEBOOKS_DIR / name
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb_content, f, indent=2)
    print(f"Generated {out_file}")

print("All 6 Jupyter notebooks generated successfully.")
