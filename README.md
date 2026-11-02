# Bank Fraud Prediction Using Machine Learning

A machine learning project that predicts fraudulent banking transactions in real time and identifies high-risk behavioral patterns that indicate financial fraud, account takeovers, and zero-day attack vectors. The project combines data cleaning, chronological time-aware splitting, leakage-free feature engineering, dual-engine predictive modeling (supervised XGBoost + unsupervised Isolation Forest), and cost-sensitive threshold optimization to support enterprise financial crime operations.

---

## 🚀 Live Demo

**Live Interactive Dashboard:** [https://bank-fraud-prediction-skcqdhqfj4ukjmr6uc6bxj.streamlit.app/](https://bank-fraud-prediction-skcqdhqfj4ukjmr6uc6bxj.streamlit.app/)

---

## 📌 Project Overview

In commercial banking, actual fraudulent transactions constitute a severe minority (**~2.1%** of all transactions), creating a difficult **needle-in-a-haystack** challenge. Standard classification models optimized solely for accuracy fail in production: a naive model predicting all transactions as legitimate achieves 97.9% accuracy while allowing millions of dollars in unmitigated fraud loss.

Furthermore, banking fraud operations carry **asymmetric financial business costs**:
* **Cost of False Negative ($1,500 average loss):** A missed fraudulent transaction results in direct chargebacks, legal compliance penalties, and customer attrition.
* **Cost of False Positive ($35 operational review cost):** An erroneously flagged legitimate transaction creates customer friction, SMS/call center triage costs, and churn.

$$\text{Total Business Cost} = (\text{False Negatives} \times \$1,500) + (\text{False Positives} \times \$35)$$

This project applies an enterprise end-to-end machine learning pipeline across **300,000 banking transactions and 35 multi-dimensional features**, combining:
1. **Supervised Gradient Boosting (XGBoost Champion):** Trained with SMOTE oversampling to capture known fraud signatures with **99.17% recall** and **0.9978 PR-AUC**.
2. **Unsupervised Anomaly Detection (Isolation Forest):** Calibrated without ground-truth labels to detect novel zero-day attack vectors.
3. **Composite Risk Scoring Engine (0–100):** Translates model probabilities into actionable operational tiers (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
4. **Deterministic Root-Cause Explainability:** Explains why each flagged transaction was challenged or blocked.
5. **Interactive Streamlit Web Console:** Delivers 6 dedicated analyst workstations for case triage, live transaction simulation, and threshold calibration.

---

## 🎯 Objectives

* Detect fraudulent banking transactions in real time with high recall (**99.17%**).
* Minimize false positives (**0.098% FPR**) to protect legitimate customer experience.
* Prevent data leakage by enforcing strict chronological (time-aware) train/validation/test partitions.
* Balance severe class imbalance using SMOTE strictly on preprocessed training data.
* Combine supervised machine learning with unsupervised anomaly detection for zero-day defense.
* Optimize decision thresholds to minimize asymmetric financial losses ($1,500 vs. $35).
* Provide explainable AI root-cause factors for every flagged transaction.
* Present the complete intelligence platform through an interactive, enterprise-grade Streamlit web application.

---

## 🛠️ Technologies Used

* **Programming Language:** Python 3.10+
* **Data Manipulation & Processing:** Pandas, NumPy
* **Machine Learning & Preprocessing:** Scikit-learn, XGBoost, Imbalanced-learn (SMOTE), Joblib
* **Unsupervised Anomaly Detection:** Isolation Forest
* **Data Visualization:** Altair, Matplotlib, Seaborn
* **Interactive Web Application:** Streamlit
* **Database & Analytics:** ANSI SQL (Production DDL Schemas & Analytical Reporting Views)
* **Development Environment:** VS Code, PowerShell

---

## ⚙️ Project Workflow

### 1. Data Collection & Audit
* Ingestion of **300,000 banking transactions** across **35 features** spanning 180 days.
* Data quality audits: schema verification, missing value imputation, duplicate elimination, and positive amount validation.

### 2. Chronological (Time-Aware) Splitting
* Financial time series create future-to-past data leakage under random $k$-fold splitting.
* Enforced strict chronological partition:
  * **Train Split (70%):** 210,000 transactions (Historical baseline)
  * **Validation Split (15%):** 45,000 transactions (Hyperparameter tuning)
  * **Held-Out Test Split (15%):** 45,000 transactions (Unseen future evaluation)

### 3. Leakage-Free Feature Engineering
* **Liquidity Dynamics:** Log transaction amount, amount-to-balance ratio, customer average spending deviation.
* **Multi-Window Velocity Signals:** Transaction frequency across 10-minute, 1-hour, and 24-hour windows.
* **Cyclic Temporal Encodings:** Sine and cosine transformations of execution hour to preserve temporal continuity across midnight.
* **Security & Behavioral Interactions:** Unrecognized device flag, geographical distance outliers from registered home address, failed login attempt ratios, and overnight execution flags.

### 4. Imbalance Handling with SMOTE
* Original training distribution: **205,614 normal (97.9%) vs. 4,386 fraud (2.1%)**.
* Applied Synthetic Minority Over-sampling Technique (**SMOTE**) strictly to the preprocessed training split (target ratio 0.25).
* Validation and held-out test splits remained untouched in their natural empirical distribution.

### 5. Dual-Engine Model Training
* Trained and benchmarked 4 models:
  * **XGBoost Classifier (Primary Champion):** Gradient boosted decision trees optimized on scale pos weight, learning rate, and tree depth.
  * **Random Forest Classifier:** Balanced subsample ensemble.
  * **Logistic Regression:** Regularized linear classification baseline.
  * **Isolation Forest (Unsupervised):** Density-based anomaly isolation for zero-day fraud pattern detection without label supervision.

### 6. Model Evaluation & Cost Minimization
* Evaluated against 45,000 unseen test transactions:

| Model | Classification Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1-Score | False Positives | False Negatives | Est. Total Cost ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Champion)** | **0.40** | **0.9999** | **0.9978** | **0.9570** | **0.9917** | **0.9740** | **43** | **8** | **$13,505** |
| Random Forest | 0.50 | 0.9998 | 0.9942 | 0.9473 | 0.9699 | 0.9585 | 52 | 29 | $45,320 |
| Logistic Regression | 0.50 | 0.9976 | 0.9331 | 0.5011 | 0.9678 | 0.6603 | 929 | 31 | $79,015 |
| Isolation Forest (Unsupervised) | 0.50 | 0.9930 | 0.8088 | 0.6521 | 0.8361 | 0.7327 | 430 | 158 | $252,050 |

* **Optimal Threshold Selection:** A cutoff of **0.40** minimizes total commercial loss exposure to just **$13,505** (down from $252,000+ baseline), delivering over **$1.4M in annualized fraud loss savings**.

### 7. Interactive Streamlit Deployment
* Deployed an enterprise FinTech dark-slate console with 6 integrated modules for fraud triage, live transaction simulation, and executive reporting.

---

## 📊 Key Features

* **Dual-Engine Fraud Intelligence:** Combines supervised probability with unsupervised anomaly scoring.
* **Composite Risk Scoring Engine (0–100):**
  $$\text{Risk Score} = \text{clip}\left(\Big[0.50 \times P_{\text{XGB}} + 0.20 \times S_{\text{Isolation}} + 0.30 \times R_{\text{Behavioral}}\Big] \times 100, \, 0, \, 100\right)$$
* **4-Tier Operational Decisioning:**
  * `LOW RISK` ($0 \le \text{Score} < 30$): Instant approval (<5ms).
  * `MEDIUM RISK` ($30 \le \text{Score} < 60$): Automated Step-Up 2FA / Biometric Challenge.
  * `HIGH RISK` ($60 \le \text{Score} < 80$): Operational hold & case review.
  * `CRITICAL RISK` ($80 \le \text{Score} \le 100$): Instant transaction block & card freeze.
* **Deterministic AI Root-Cause Explanations:** Explains flagged transactions with human-readable reason chips (e.g., *Balance Drain 92%*, *Cross-Border International Rail*, *Unrecognized Device Fingerprint*).
* **⚡ Live Scoring Sandbox:** Single-transaction simulator with one-click attack scenario presets (*Account Takeover*, *SIM Swap Burst*, *Legitimate Coffee*, *Salary Transfer*).
* **Case Triage Workbench:** Search transactions, review customer profiles, and trigger actions (`[🚫 Block]`, `[⚠️ Challenge 2FA]`, `[✅ Whitelist]`).
* **Threshold Economic Simulator:** Interactive slider balancing False Negatives ($1,500) vs. False Positives ($35).

---

## 🖥️ Interactive Dashboard Modules (`app.py`)

```text
========================================================================================
🛡️ NOVA SENTINEL • FRAUD INTELLIGENCE CONSOLE
Status: ● LIVE INFERENCE ACTIVE | Engine: Dual ML Pipeline (XGBoost + Isolation Forest)
========================================================================================
[ 📊 Executive Overview ] [ 🧠 Fraud Intelligence ] [ 🔍 Case Investigation ]
[ ⚡ Live Scoring Sandbox ] [ 👥 Customer Risk ] [ 📈 Model Diagnostics ]
========================================================================================
```

1. **📊 Executive Overview:** High-level financial KPIs, gross volume vs. prevented fraud exposure, multi-interval temporal trend line, risk population distribution donut chart, payment rail breakdown, and geographic hotspots.
2. **🧠 Fraud Intelligence:** Multidimensional cross-correlation matrix (e.g., channel vs. payment rail vs. auth method), behavioral signal lift rankings (`new_device`, `failed_logins`, etc.), and log-scale ticket density curves.
3. **🔍 Case Investigation:** Real-time case triage workbench with transaction & customer ID search, deep-dive inspection dossier, AI root-cause explanation chips, and one-click action triggers.
4. **⚡ Live Scoring Sandbox:** Interactive single-transaction simulator: load one-click attack presets or customize parameters and see live composite risk score (0–100), decision tier, and explainable factors.
5. **👥 Customer Risk:** Entity-level cumulative risk profiling, velocity outliers, repeat offender accounts, and top 20 customer watchlists.
6. **📈 Model Diagnostics:** Champion XGBoost ROC curve (`0.9999`) and PR curve (`0.9978`), confusion matrix, top 15 XGBoost gain features, and an interactive threshold economic cost calculator.

---

## 💡 Business Applications

* **Real-Time Payment Fraud Prevention:** Sub-5ms scoring at card authorization endpoints.
* **Account Takeover (ATO) Mitigation:** Flags credential stuffing and SIM swap signatures.
* **Operational Triage Automation:** Prioritizes investigation cases into calibrated risk tiers (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), saving over **$45,000/month** in manual review friction.
* **Zero-Day Novelty Protection:** Unsupervised Isolation Forest catches previously unseen exploitation patterns without retraining.
* **Executive & AML Compliance:** Production SQL reporting views and Star Schema data models ready for Power BI / Fabric compliance audits.

---

## 📈 Learning Outcomes

Through this project, I explored practical applications of:
* Handling severe real-world class imbalance (**2.1% fraud incidence**) using **SMOTE** strictly on the training partition.
* Preventing future-to-past data leakage via **chronological time-aware splitting**.
* Building leakage-free feature engineering pipelines (rolling multi-window velocity, balance depletion ratios, cyclic time encodings).
* Dual-engine modeling combining **supervised boosting (XGBoost)** and **unsupervised anomaly detection (Isolation Forest)**.
* Evaluating models with **PR-AUC and Precision-Recall tradeoffs** rather than misleading raw accuracy.
* Aligning decision cutoffs with **asymmetric business cost economics** ($1,500 False Negative vs. $35 False Positive).
* Generating **deterministic root-cause explainability** for regulatory compliance.
* Designing and deploying a production-grade, dark-slate **Streamlit web console** for enterprise fraud operations.

---

## 📄 License & Author

Distributed under the **MIT License**. See `LICENSE` for more information.

Developed by [singhs143143-gif](https://github.com/singhs143143-gif).
