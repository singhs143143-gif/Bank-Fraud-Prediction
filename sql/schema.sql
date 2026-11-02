-- =============================================================================
-- SQL SCHEMA: INTELLIGENT BANKING TRANSACTION & FRAUD DETECTION PLATFORM
-- Database Engine: ANSI SQL / PostgreSQL / SQL Server / MySQL 8.0+ Compatible
-- =============================================================================

-- Drop tables if re-initializing schema
DROP TABLE IF EXISTS fraud_alerts;
DROP TABLE IF EXISTS fraud_predictions;
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS customer_risk_profiles;
DROP TABLE IF EXISTS customers;

-- -----------------------------------------------------------------------------
-- 1. CUSTOMERS TABLE
-- -----------------------------------------------------------------------------
CREATE TABLE customers (
    customer_id                VARCHAR(32) PRIMARY KEY,
    customer_age               INTEGER NOT NULL CHECK (customer_age >= 18),
    customer_income            DECIMAL(14, 2) NOT NULL CHECK (customer_income >= 0),
    account_balance            DECIMAL(14, 2) NOT NULL CHECK (account_balance >= 0),
    customer_location          VARCHAR(100) NOT NULL,
    primary_device_id          VARCHAR(64),
    average_transaction_amount DECIMAL(12, 2) NOT NULL CHECK (average_transaction_amount >= 0),
    transaction_frequency      INTEGER NOT NULL DEFAULT 15,
    account_age_days           INTEGER NOT NULL DEFAULT 30,
    created_at                 TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    account_status             VARCHAR(20) NOT NULL DEFAULT 'ACTIVE' CHECK (account_status IN ('ACTIVE', 'RESTRICTED', 'FROZEN', 'CLOSED'))
);

CREATE INDEX idx_cust_location ON customers(customer_location);
CREATE INDEX idx_cust_balance ON customers(account_balance);

-- -----------------------------------------------------------------------------
-- 2. TRANSACTIONS TABLE
-- -----------------------------------------------------------------------------
CREATE TABLE transactions (
    transaction_id              VARCHAR(32) PRIMARY KEY,
    customer_id                 VARCHAR(32) NOT NULL,
    transaction_timestamp       TIMESTAMP NOT NULL,
    transaction_amount          DECIMAL(14, 2) NOT NULL CHECK (transaction_amount > 0),
    transaction_type            VARCHAR(30) NOT NULL CHECK (transaction_type IN ('PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN')),
    merchant_category           VARCHAR(60) NOT NULL,
    merchant_id                 VARCHAR(32) NOT NULL,
    account_age_days            INTEGER NOT NULL,
    customer_age                INTEGER NOT NULL,
    customer_income             DECIMAL(14, 2) NOT NULL,
    account_balance             DECIMAL(14, 2) NOT NULL,
    transaction_location        VARCHAR(100) NOT NULL,
    customer_location           VARCHAR(100) NOT NULL,
    device_id                   VARCHAR(64) NOT NULL,
    device_type                 VARCHAR(40) NOT NULL,
    payment_method              VARCHAR(40) NOT NULL,
    authentication_method       VARCHAR(40) NOT NULL,
    ip_address                  VARCHAR(45) NOT NULL,
    previous_transaction_amount DECIMAL(14, 2),
    transaction_frequency       INTEGER NOT NULL,
    failed_transaction_count    INTEGER NOT NULL DEFAULT 0,
    login_attempts              INTEGER NOT NULL DEFAULT 1,
    international_transaction   SMALLINT NOT NULL DEFAULT 0 CHECK (international_transaction IN (0, 1)),
    weekend_transaction         SMALLINT NOT NULL DEFAULT 0 CHECK (weekend_transaction IN (0, 1)),
    night_transaction           SMALLINT NOT NULL DEFAULT 0 CHECK (night_transaction IN (0, 1)),
    new_device                  SMALLINT NOT NULL DEFAULT 0 CHECK (new_device IN (0, 1)),
    new_location                SMALLINT NOT NULL DEFAULT 0 CHECK (new_location IN (0, 1)),
    distance_from_home          DECIMAL(10, 2) NOT NULL DEFAULT 0.0,
    transaction_velocity        DECIMAL(8, 2) NOT NULL DEFAULT 1.0,
    customer_transaction_count  INTEGER NOT NULL DEFAULT 1,
    average_transaction_amount  DECIMAL(12, 2) NOT NULL,
    transaction_hour            SMALLINT NOT NULL CHECK (transaction_hour BETWEEN 0 AND 23),
    transaction_day             SMALLINT NOT NULL CHECK (transaction_day BETWEEN 1 AND 31),
    transaction_month           SMALLINT NOT NULL CHECK (transaction_month BETWEEN 1 AND 12),
    fraud_label                 SMALLINT NOT NULL DEFAULT 0 CHECK (fraud_label IN (0, 1)),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

CREATE INDEX idx_txn_timestamp ON transactions(transaction_timestamp);
CREATE INDEX idx_txn_cust_time ON transactions(customer_id, transaction_timestamp);
CREATE INDEX idx_txn_fraud ON transactions(fraud_label);
CREATE INDEX idx_txn_type ON transactions(transaction_type);
CREATE INDEX idx_txn_merchant_cat ON transactions(merchant_category);

-- -----------------------------------------------------------------------------
-- 3. FRAUD PREDICTIONS TABLE (ML & Anomaly Detection Pipeline Output)
-- -----------------------------------------------------------------------------
CREATE TABLE fraud_predictions (
    prediction_id        VARCHAR(32) PRIMARY KEY,
    transaction_id       VARCHAR(32) NOT NULL UNIQUE,
    customer_id          VARCHAR(32) NOT NULL,
    fraud_probability    DECIMAL(6, 4) NOT NULL CHECK (fraud_probability BETWEEN 0.0 AND 1.0),
    anomaly_score        DECIMAL(6, 4) NOT NULL CHECK (anomaly_score BETWEEN 0.0 AND 1.0),
    risk_score           SMALLINT NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    risk_level           VARCHAR(10) NOT NULL CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH')),
    recommended_action   VARCHAR(40) NOT NULL CHECK (recommended_action IN ('ALLOW', 'VERIFY / CHALLENGE', 'BLOCK / INVESTIGATE')),
    model_version        VARCHAR(30) NOT NULL DEFAULT 'xgboost_v2.0',
    primary_reasons      TEXT,
    predicted_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

CREATE INDEX idx_pred_risk_level ON fraud_predictions(risk_level);
CREATE INDEX idx_pred_risk_score ON fraud_predictions(risk_score);

-- -----------------------------------------------------------------------------
-- 4. FRAUD ALERTS TABLE (Operational Investigation Triage Queue)
-- -----------------------------------------------------------------------------
CREATE TABLE fraud_alerts (
    alert_id            VARCHAR(32) PRIMARY KEY,
    transaction_id      VARCHAR(32) NOT NULL,
    customer_id         VARCHAR(32) NOT NULL,
    risk_score          SMALLINT NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    risk_level          VARCHAR(10) NOT NULL CHECK (risk_level IN ('MEDIUM', 'HIGH')),
    fraud_probability   DECIMAL(6, 4) NOT NULL,
    anomaly_score       DECIMAL(6, 4) NOT NULL,
    alert_reason        TEXT NOT NULL,
    recommended_action  VARCHAR(40) NOT NULL,
    assigned_analyst    VARCHAR(60) DEFAULT 'UNASSIGNED',
    status              VARCHAR(25) NOT NULL DEFAULT 'NEW' CHECK (status IN ('NEW', 'UNDER_REVIEW', 'CONFIRMED_FRAUD', 'FALSE_POSITIVE', 'RESOLVED')),
    resolution_notes    TEXT,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at         TIMESTAMP,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

CREATE INDEX idx_alert_status ON fraud_alerts(status);
CREATE INDEX idx_alert_created ON fraud_alerts(created_at);
CREATE INDEX idx_alert_analyst ON fraud_alerts(assigned_analyst);

-- -----------------------------------------------------------------------------
-- 5. CUSTOMER RISK PROFILES TABLE (Aggregated Customer Intelligence)
-- -----------------------------------------------------------------------------
CREATE TABLE customer_risk_profiles (
    customer_id                VARCHAR(32) PRIMARY KEY,
    total_lifetime_txns        INTEGER NOT NULL DEFAULT 0,
    total_lifetime_spend       DECIMAL(14, 2) NOT NULL DEFAULT 0.00,
    total_fraud_count          INTEGER NOT NULL DEFAULT 0,
    total_alerts_count         INTEGER NOT NULL DEFAULT 0,
    customer_risk_index        SMALLINT NOT NULL DEFAULT 10 CHECK (customer_risk_index BETWEEN 0 AND 100),
    risk_segment               VARCHAR(20) NOT NULL DEFAULT 'STANDARD' CHECK (risk_segment IN ('LOW_RISK', 'STANDARD', 'WATCHLIST', 'CRITICAL_RISK')),
    last_known_device_id       VARCHAR(64),
    last_known_ip_address      VARCHAR(45),
    last_assessed_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

CREATE INDEX idx_cust_risk_segment ON customer_risk_profiles(risk_segment);
CREATE INDEX idx_cust_risk_index ON customer_risk_profiles(customer_risk_index);
