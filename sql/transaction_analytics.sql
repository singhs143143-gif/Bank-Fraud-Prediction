-- =============================================================================
-- SQL ANALYTICS: TRANSACTION PATTERNS & POWER BI CONSUMPTION VIEWS
-- System: Intelligent Banking Transaction & Fraud Detection Platform
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. SUSPICIOUS NEW-DEVICE ACTIVITY
-- Targets transactions executed from unrecognized devices with elevated fraud risk
-- -----------------------------------------------------------------------------
SELECT
    t.transaction_id,
    t.customer_id,
    t.device_id,
    t.device_type,
    t.transaction_timestamp,
    t.transaction_amount,
    t.failed_transaction_count,
    t.fraud_label,
    p.risk_score,
    p.risk_level
FROM transactions t
LEFT JOIN fraud_predictions p ON t.transaction_id = p.transaction_id
WHERE t.new_device = 1
  AND (t.fraud_label = 1 OR p.risk_score >= 70)
ORDER BY t.transaction_amount DESC
LIMIT 100;

-- -----------------------------------------------------------------------------
-- 2. SUSPICIOUS CROSS-BORDER / INTERNATIONAL TRANSACTIONS
-- Tracks high-distance foreign transactions coupled with high amounts
-- -----------------------------------------------------------------------------
SELECT
    t.transaction_id,
    t.customer_id,
    t.customer_location AS domestic_home_base,
    t.transaction_location AS overseas_txn_location,
    t.distance_from_home AS distance_km,
    t.transaction_amount,
    t.payment_method,
    t.fraud_label,
    p.risk_score,
    p.recommended_action
FROM transactions t
LEFT JOIN fraud_predictions p ON t.transaction_id = p.transaction_id
WHERE t.international_transaction = 1
  AND t.distance_from_home > 3000
ORDER BY t.transaction_amount DESC
LIMIT 100;

-- -----------------------------------------------------------------------------
-- 3. BALANCE DRAIN & BUST-OUT TRANSACTION PATTERNS
-- Outgoing transfers or cash-outs draining > 80% of total balance
-- -----------------------------------------------------------------------------
SELECT
    t.transaction_id,
    t.customer_id,
    t.transaction_type,
    t.transaction_amount,
    t.account_balance,
    ROUND(t.transaction_amount / NULLIF(t.account_balance, 0) * 100, 2) AS balance_drain_pct,
    t.night_transaction,
    t.fraud_label,
    p.risk_score
FROM transactions t
LEFT JOIN fraud_predictions p ON t.transaction_id = p.transaction_id
WHERE t.transaction_type IN ('TRANSFER', 'CASH_OUT')
  AND t.account_balance > 0
  AND (t.transaction_amount / t.account_balance) >= 0.80
ORDER BY t.transaction_amount DESC
LIMIT 100;

-- =============================================================================
-- POWER BI REPORTING VIEWS
-- Optimized consumption views for executive dashboards and analyst consoles
-- =============================================================================

-- -----------------------------------------------------------------------------
-- VIEW 1: v_executive_fraud_kpis
-- Feeds Page 1: Executive Fraud Overview
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_executive_fraud_kpis AS
SELECT
    CAST(t.transaction_timestamp AS DATE) AS transaction_date,
    t.transaction_type,
    t.merchant_category,
    t.transaction_location,
    COUNT(t.transaction_id) AS total_transactions,
    ROUND(SUM(t.transaction_amount), 2) AS total_volume_usd,
    SUM(t.fraud_label) AS actual_fraud_count,
    ROUND(SUM(CASE WHEN t.fraud_label = 1 THEN t.transaction_amount ELSE 0 END), 2) AS actual_fraud_amount_usd,
    COUNT(CASE WHEN p.risk_level = 'HIGH' THEN 1 END) AS high_risk_transactions,
    COUNT(CASE WHEN p.risk_level = 'MEDIUM' THEN 1 END) AS medium_risk_transactions,
    COUNT(CASE WHEN p.risk_level = 'LOW' THEN 1 END) AS low_risk_transactions,
    COUNT(CASE WHEN p.risk_level = 'HIGH' AND t.fraud_label = 1 THEN 1 END) AS true_fraud_caught,
    COUNT(CASE WHEN p.risk_level = 'HIGH' AND t.fraud_label = 0 THEN 1 END) AS false_positives_blocked
FROM transactions t
LEFT JOIN fraud_predictions p ON t.transaction_id = p.transaction_id
GROUP BY 
    CAST(t.transaction_timestamp AS DATE),
    t.transaction_type,
    t.merchant_category,
    t.transaction_location;

-- -----------------------------------------------------------------------------
-- VIEW 2: v_customer_risk_intelligence
-- Feeds Page 2: Customer Risk Intelligence
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_customer_risk_intelligence AS
SELECT
    c.customer_id,
    c.customer_age,
    c.customer_income,
    c.account_balance,
    c.customer_location,
    c.average_transaction_amount,
    c.transaction_frequency,
    COUNT(t.transaction_id) AS recorded_transactions,
    ROUND(SUM(t.transaction_amount), 2) AS total_spend_usd,
    SUM(t.fraud_label) AS total_fraud_events,
    SUM(t.failed_transaction_count) AS total_failed_logins,
    ROUND(AVG(COALESCE(p.risk_score, 10)), 1) AS avg_transaction_risk_score,
    MAX(COALESCE(p.risk_score, 0)) AS peak_transaction_risk_score,
    CASE
        WHEN SUM(t.fraud_label) > 0 OR MAX(COALESCE(p.risk_score, 0)) >= 80 THEN 'CRITICAL RISK'
        WHEN MAX(COALESCE(p.risk_score, 0)) >= 50 OR SUM(t.failed_transaction_count) >= 3 THEN 'WATCHLIST'
        ELSE 'STANDARD'
    END AS customer_risk_tier
FROM customers c
LEFT JOIN transactions t ON c.customer_id = t.customer_id
LEFT JOIN fraud_predictions p ON t.transaction_id = p.transaction_id
GROUP BY 
    c.customer_id,
    c.customer_age,
    c.customer_income,
    c.account_balance,
    c.customer_location,
    c.average_transaction_amount,
    c.transaction_frequency;

-- -----------------------------------------------------------------------------
-- VIEW 3: v_transaction_investigation_feed
-- Feeds Page 3: Transaction Investigation Console
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_transaction_investigation_feed AS
SELECT
    t.transaction_id,
    t.customer_id,
    t.transaction_timestamp,
    t.transaction_amount,
    t.transaction_type,
    t.merchant_category,
    t.merchant_id,
    t.transaction_location,
    t.device_id,
    t.device_type,
    t.payment_method,
    t.authentication_method,
    t.ip_address,
    t.distance_from_home,
    t.transaction_velocity,
    t.failed_transaction_count,
    t.new_device,
    t.new_location,
    t.international_transaction,
    t.fraud_label AS actual_fraud_status,
    COALESCE(p.fraud_probability, 0.0) AS fraud_probability,
    COALESCE(p.anomaly_score, 0.0) AS anomaly_score,
    COALESCE(p.risk_score, 0) AS risk_score,
    COALESCE(p.risk_level, 'LOW') AS risk_level,
    COALESCE(p.recommended_action, 'ALLOW') AS recommended_action,
    p.primary_reasons AS explanation_notes,
    a.alert_id,
    COALESCE(a.status, 'NO_ALERT') AS alert_status
FROM transactions t
LEFT JOIN fraud_predictions p ON t.transaction_id = p.transaction_id
LEFT JOIN fraud_alerts a ON t.transaction_id = a.transaction_id;

-- -----------------------------------------------------------------------------
-- VIEW 4: v_model_performance_summary
-- Feeds Page 4: ML Model Performance & Threshold Tuning
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_model_performance_summary AS
SELECT
    p.model_version,
    COUNT(*) AS total_scored,
    SUM(CASE WHEN t.fraud_label = 1 AND p.risk_level = 'HIGH' THEN 1 ELSE 0 END) AS true_positives,
    SUM(CASE WHEN t.fraud_label = 0 AND p.risk_level = 'HIGH' THEN 1 ELSE 0 END) AS false_positives,
    SUM(CASE WHEN t.fraud_label = 1 AND p.risk_level != 'HIGH' THEN 1 ELSE 0 END) AS false_negatives,
    SUM(CASE WHEN t.fraud_label = 0 AND p.risk_level != 'HIGH' THEN 1 ELSE 0 END) AS true_negatives,
    ROUND(AVG(p.fraud_probability), 4) AS mean_predicted_probability,
    ROUND(AVG(p.anomaly_score), 4) AS mean_anomaly_score,
    ROUND(AVG(p.risk_score), 2) AS mean_risk_score
FROM fraud_predictions p
JOIN transactions t ON p.transaction_id = t.transaction_id
GROUP BY p.model_version;
