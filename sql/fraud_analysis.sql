-- =============================================================================
-- SQL ANALYTICS: FRAUD DETECTION & TRANSACTION PATTERN ANALYSIS
-- System: Intelligent Banking Transaction & Fraud Detection Platform
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. OVERALL EXECUTIVE FRAUD KPIs
-- Calculates total volume, fraud count, net fraud losses, and overall fraud incidence
-- -----------------------------------------------------------------------------
SELECT
    COUNT(*) AS total_transactions,
    ROUND(SUM(transaction_amount), 2) AS total_transaction_volume_usd,
    SUM(fraud_label) AS total_fraud_transactions,
    ROUND(SUM(CASE WHEN fraud_label = 1 THEN transaction_amount ELSE 0 END), 2) AS total_fraud_loss_usd,
    ROUND(AVG(fraud_label) * 100, 3) AS fraud_rate_percentage,
    ROUND(AVG(transaction_amount), 2) AS avg_ticket_size_usd,
    ROUND(AVG(CASE WHEN fraud_label = 1 THEN transaction_amount END), 2) AS avg_fraud_ticket_size_usd
FROM transactions;

-- -----------------------------------------------------------------------------
-- 2. FRAUD BREAKDOWN BY TRANSACTION TYPE
-- Identifies high-risk channels (e.g., Wire Transfers vs Point of Sale)
-- -----------------------------------------------------------------------------
SELECT
    transaction_type,
    COUNT(*) AS total_transactions,
    ROUND(SUM(transaction_amount), 2) AS total_amount,
    SUM(fraud_label) AS fraud_transactions,
    ROUND(SUM(CASE WHEN fraud_label = 1 THEN transaction_amount ELSE 0 END), 2) AS fraud_amount,
    ROUND(AVG(fraud_label) * 100, 2) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN fraud_label = 1 THEN transaction_amount ELSE 0 END) / 
          NULLIF(SUM(transaction_amount), 0) * 100, 2) AS fraud_volume_share_pct
FROM transactions
GROUP BY transaction_type
ORDER BY fraud_amount DESC;

-- -----------------------------------------------------------------------------
-- 3. FRAUD BREAKDOWN BY MERCHANT CATEGORY
-- Pinpoints high-target merchant sectors (e.g., Luxury Goods, Electronics)
-- -----------------------------------------------------------------------------
SELECT
    merchant_category,
    COUNT(*) AS total_transactions,
    SUM(fraud_label) AS fraud_transactions,
    ROUND(SUM(CASE WHEN fraud_label = 1 THEN transaction_amount ELSE 0 END), 2) AS fraud_loss_usd,
    ROUND(AVG(fraud_label) * 100, 2) AS fraud_incidence_pct,
    ROUND(AVG(CASE WHEN fraud_label = 1 THEN transaction_amount END), 2) AS avg_fraud_amount
FROM transactions
GROUP BY merchant_category
ORDER BY fraud_loss_usd DESC;

-- -----------------------------------------------------------------------------
-- 4. FRAUD BREAKDOWN BY PAYMENT METHOD
-- Evaluates vulnerability of different instrument channels
-- -----------------------------------------------------------------------------
SELECT
    payment_method,
    COUNT(*) AS total_transactions,
    SUM(fraud_label) AS fraud_count,
    ROUND(SUM(CASE WHEN fraud_label = 1 THEN transaction_amount ELSE 0 END), 2) AS total_fraud_loss,
    ROUND(AVG(fraud_label) * 100, 3) AS fraud_rate_pct
FROM transactions
GROUP BY payment_method
ORDER BY fraud_count DESC;

-- -----------------------------------------------------------------------------
-- 5. GEOGRAPHICAL FRAUD EXPOSURE (TRANSACTION LOCATION)
-- Highlights geographic hotspots and regional cross-border risk
-- -----------------------------------------------------------------------------
SELECT
    transaction_location,
    COUNT(*) AS total_txns,
    SUM(fraud_label) AS fraud_count,
    ROUND(SUM(CASE WHEN fraud_label = 1 THEN transaction_amount ELSE 0 END), 2) AS fraud_exposure_usd,
    ROUND(AVG(fraud_label) * 100, 2) AS fraud_rate_pct,
    SUM(international_transaction) AS international_txn_count
FROM transactions
GROUP BY transaction_location
ORDER BY fraud_exposure_usd DESC;

-- -----------------------------------------------------------------------------
-- 6. DIURNAL FRAUD PATTERN (HOUR OF DAY ANALYSIS)
-- Evaluates nocturnal/off-peak fraud attacks vs normal daytime banking hours
-- -----------------------------------------------------------------------------
SELECT
    transaction_hour,
    COUNT(*) AS total_txns,
    SUM(fraud_label) AS fraud_count,
    ROUND(AVG(fraud_label) * 100, 2) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN fraud_label = 1 THEN transaction_amount ELSE 0 END), 2) AS total_fraud_amount,
    CASE 
        WHEN transaction_hour BETWEEN 0 AND 5 THEN 'Overnight Graveyard (00:00 - 05:00)'
        WHEN transaction_hour BETWEEN 6 AND 11 THEN 'Morning Peak (06:00 - 11:00)'
        WHEN transaction_hour BETWEEN 12 AND 17 THEN 'Afternoon Standard (12:00 - 17:00)'
        ELSE 'Evening (18:00 - 23:00)'
    END AS daypart_category
FROM transactions
GROUP BY transaction_hour
ORDER BY transaction_hour ASC;

-- -----------------------------------------------------------------------------
-- 7. DAILY CHRONOLOGICAL FRAUD TREND (RUNNING TOTAL & 7-DAY MOVING AVERAGE)
-- -----------------------------------------------------------------------------
WITH daily_metrics AS (
    SELECT
        CAST(transaction_timestamp AS DATE) AS txn_date,
        COUNT(*) AS daily_total_txns,
        SUM(fraud_label) AS daily_fraud_txns,
        ROUND(SUM(CASE WHEN fraud_label = 1 THEN transaction_amount ELSE 0 END), 2) AS daily_fraud_amount,
        ROUND(AVG(fraud_label) * 100, 3) AS daily_fraud_rate_pct
    FROM transactions
    GROUP BY CAST(transaction_timestamp AS DATE)
)
SELECT
    txn_date,
    daily_total_txns,
    daily_fraud_txns,
    daily_fraud_amount,
    daily_fraud_rate_pct,
    AVG(daily_fraud_rate_pct) OVER (
        ORDER BY txn_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_avg_fraud_rate
FROM daily_metrics
ORDER BY txn_date ASC;

-- -----------------------------------------------------------------------------
-- 8. WEEKLY FRAUD EXPOSURE & ATTACK CLUSTERING
-- -----------------------------------------------------------------------------
SELECT
    DATE_TRUNC('week', transaction_timestamp) AS week_start,
    COUNT(*) AS total_weekly_txns,
    SUM(fraud_label) AS weekly_fraud_txns,
    ROUND(SUM(CASE WHEN fraud_label = 1 THEN transaction_amount ELSE 0 END), 2) AS weekly_fraud_amount,
    ROUND(AVG(fraud_label) * 100, 2) AS weekly_fraud_rate_pct
FROM transactions
GROUP BY DATE_TRUNC('week', transaction_timestamp)
ORDER BY week_start ASC;
