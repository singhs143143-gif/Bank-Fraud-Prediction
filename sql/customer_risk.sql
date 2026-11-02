-- =============================================================================
-- SQL ANALYTICS: CUSTOMER RISK INTELLIGENCE & ENTITY PROFILING
-- System: Intelligent Banking Transaction & Fraud Detection Platform
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. TOP FRAUD-IMPACTED CUSTOMERS (BY TOTAL LOSS VALUE)
-- Identifies accounts targeted by severe account takeover or balance drain
-- -----------------------------------------------------------------------------
SELECT
    t.customer_id,
    c.customer_location,
    c.account_balance,
    COUNT(t.transaction_id) AS total_transactions,
    SUM(t.fraud_label) AS total_fraud_transactions,
    ROUND(SUM(CASE WHEN t.fraud_label = 1 THEN t.transaction_amount ELSE 0 END), 2) AS total_fraud_loss_usd,
    ROUND(AVG(t.fraud_label) * 100, 2) AS customer_fraud_rate_pct,
    MAX(t.transaction_timestamp) AS latest_fraud_timestamp
FROM transactions t
JOIN customers c ON t.customer_id = c.customer_id
GROUP BY t.customer_id, c.customer_location, c.account_balance
HAVING SUM(t.fraud_label) > 0
ORDER BY total_fraud_loss_usd DESC
LIMIT 50;

-- -----------------------------------------------------------------------------
-- 2. HIGH-RISK MERCHANTS (BY FRAUD VOLUME & INCIDENCE RATE)
-- Identifies compromised or high-collusion merchant terminals
-- -----------------------------------------------------------------------------
SELECT
    merchant_id,
    merchant_category,
    COUNT(*) AS total_transactions_processed,
    SUM(fraud_label) AS total_fraud_transactions,
    ROUND(SUM(CASE WHEN fraud_label = 1 THEN transaction_amount ELSE 0 END), 2) AS total_fraud_amount_usd,
    ROUND(AVG(fraud_label) * 100, 2) AS merchant_fraud_rate_pct,
    COUNT(DISTINCT customer_id) AS unique_customers_affected
FROM transactions
GROUP BY merchant_id, merchant_category
HAVING COUNT(*) >= 50
ORDER BY merchant_fraud_rate_pct DESC, total_fraud_amount_usd DESC
LIMIT 50;

-- -----------------------------------------------------------------------------
-- 3. UNUSUAL TRANSACTION VELOCITY OUTLIERS
-- Identifies customers transacting significantly above normal velocity thresholds
-- -----------------------------------------------------------------------------
SELECT
    t.customer_id,
    t.transaction_id,
    t.transaction_timestamp,
    t.transaction_velocity,
    c.transaction_frequency AS normal_monthly_frequency,
    ROUND(t.transaction_velocity / (c.transaction_frequency / 30.0), 2) AS velocity_surge_multiplier,
    t.transaction_amount,
    t.fraud_label
FROM transactions t
JOIN customers c ON t.customer_id = c.customer_id
WHERE t.transaction_velocity >= 6.0
ORDER BY velocity_surge_multiplier DESC
LIMIT 100;

-- -----------------------------------------------------------------------------
-- 4. REPEATED FAILED AUTHENTICATION / SUSPICIOUS LOGIN CLUSTERS
-- Credential stuffing or brute-force attack signatures
-- -----------------------------------------------------------------------------
SELECT
    customer_id,
    COUNT(transaction_id) AS suspicious_events_count,
    SUM(failed_transaction_count) AS total_failed_attempts,
    AVG(login_attempts) AS avg_login_attempts_per_txn,
    SUM(fraud_label) AS resulting_fraud_transactions,
    MAX(transaction_timestamp) AS last_suspicious_event
FROM transactions
WHERE failed_transaction_count >= 2 OR login_attempts >= 3
GROUP BY customer_id
ORDER BY total_failed_attempts DESC
LIMIT 50;

-- -----------------------------------------------------------------------------
-- 5. COMPREHENSIVE CUSTOMER RISK INDEX CALCULATION
-- Dynamically scores customer exposure based on behavioral deviations
-- -----------------------------------------------------------------------------
WITH customer_aggregations AS (
    SELECT
        c.customer_id,
        COUNT(t.transaction_id) AS lifetime_txns,
        SUM(t.fraud_label) AS fraud_events,
        SUM(t.failed_transaction_count) AS failed_events,
        SUM(t.new_device) AS new_device_events,
        SUM(t.new_location) AS new_location_events,
        MAX(t.transaction_amount) AS max_amount,
        c.average_transaction_amount AS avg_amt
    FROM customers c
    LEFT JOIN transactions t ON c.customer_id = t.customer_id
    GROUP BY c.customer_id, c.average_transaction_amount
)
SELECT
    customer_id,
    lifetime_txns,
    fraud_events,
    -- Normalized Risk Score Formula (0 to 100)
    LEAST(100, GREATEST(0, CAST(
        (fraud_events * 40.0) +
        (failed_events * 5.0) +
        (new_device_events * 3.0) +
        (new_location_events * 3.0) +
        (CASE WHEN max_amount > avg_amt * 3.0 THEN 15.0 ELSE 0.0 END)
    AS INTEGER))) AS dynamic_customer_risk_score,
    CASE
        WHEN fraud_events > 0 THEN 'CRITICAL_RISK'
        WHEN (failed_events >= 3 OR new_device_events >= 4) THEN 'WATCHLIST'
        WHEN lifetime_txns > 50 THEN 'STANDARD'
        ELSE 'LOW_RISK'
    END AS risk_tier
FROM customer_aggregations
ORDER BY dynamic_customer_risk_score DESC;
