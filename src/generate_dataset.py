"""
generate_dataset.py
-------------------
Generates a realistic, large-scale banking transaction dataset containing
300,000+ records and 35+ industry-standard banking features with realistic
class imbalance and non-trivial fraud mechanisms.
"""

import datetime
import logging
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("generate_dataset")

RANDOM_SEED = 42
TOTAL_RECORDS = 300_000
NUM_CUSTOMERS = 15_000
NUM_MERCHANTS = 1_200

LOCATIONS = [
    "New York, USA", "Los Angeles, USA", "Chicago, USA", "Houston, USA",
    "London, UK", "Toronto, CAN", "Sydney, AUS", "Frankfurt, GER",
    "Paris, FRA", "Singapore, SGP", "Tokyo, JPN", "Dubai, UAE"
]

DEVICE_TYPES = ["Mobile iOS", "Mobile Android", "Desktop Windows", "Desktop MacOS", "Tablet"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "Wire Transfer", "UPI / P2P", "Net Banking"]
AUTH_METHODS = ["Biometric", "PIN", "Password", "SMS OTP", "Hardware Token"]
TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]

MERCHANT_CATEGORIES = {
    "Grocery": 0.28,
    "Dining & Entertainment": 0.22,
    "Online Retail": 0.20,
    "Utilities": 0.10,
    "Electronics": 0.08,
    "Travel & Airlines": 0.05,
    "Financial Services": 0.04,
    "Luxury Goods": 0.03,
}

def generate_synthetic_bank_data(
    num_records: int = TOTAL_RECORDS,
    num_customers: int = NUM_CUSTOMERS,
    seed: int = RANDOM_SEED,
    output_path: Path = None,
    sample_path: Path = None,
) -> pd.DataFrame:
    logger.info("Starting synthetic dataset generation (%d records)...", num_records)
    np.random.seed(seed)

    # 1. Generate Customer Master Profiles
    customer_ids = [f"CUST_{i:06d}" for i in range(1, num_customers + 1)]
    cust_age = np.random.randint(18, 76, size=num_customers)
    cust_income = np.round(np.random.lognormal(mean=10.8, sigma=0.6, size=num_customers), 2)
    cust_income = np.clip(cust_income, 18000, 350000)
    cust_balance = np.round(np.random.lognormal(mean=9.2, sigma=1.1, size=num_customers), 2)
    cust_balance = np.clip(cust_balance, 100, 500000)
    cust_home_loc = np.random.choice(LOCATIONS[:6], size=num_customers) # Primary domestic locations
    cust_primary_device = [f"DEV_{np.random.randint(100000, 999999)}" for _ in range(num_customers)]
    cust_avg_txn = np.round(np.random.gamma(shape=4.0, scale=35.0, size=num_customers), 2)
    cust_txn_freq = np.random.randint(5, 65, size=num_customers) # monthly frequency
    cust_account_age = np.random.randint(30, 4000, size=num_customers)

    cust_df = pd.DataFrame({
        "customer_id": customer_ids,
        "customer_age": cust_age,
        "customer_income": cust_income,
        "account_balance": cust_balance,
        "customer_location": cust_home_loc,
        "primary_device_id": cust_primary_device,
        "average_transaction_amount": cust_avg_txn,
        "transaction_frequency": cust_txn_freq,
        "account_age_days": cust_account_age,
    })

    # 2. Generate Merchants
    merchant_ids = [f"MERCH_{i:05d}" for i in range(1, NUM_MERCHANTS + 1)]
    cats = list(MERCHANT_CATEGORIES.keys())
    cat_weights = list(MERCHANT_CATEGORIES.values())
    merch_cats = np.random.choice(cats, size=NUM_MERCHANTS, p=cat_weights)
    merch_df = pd.DataFrame({"merchant_id": merchant_ids, "merchant_category": merch_cats})

    # 3. Sample Customer Index for each transaction (Power law transaction distribution)
    # Some customers transact much more often than others
    customer_weights = np.random.pareto(a=2.5, size=num_customers)
    customer_weights /= customer_weights.sum()
    chosen_cust_indices = np.random.choice(num_customers, size=num_records, p=customer_weights)

    # 4. Generate Timestamps over a 180-day window
    start_time = datetime.datetime(2024, 1, 1, 0, 0, 0)
    # Seconds across 180 days: 180 * 86400 = 15,552,000
    random_seconds = np.sort(np.random.randint(0, 180 * 86400, size=num_records))
    timestamps = [start_time + datetime.timedelta(seconds=int(s)) for s in random_seconds]

    hours = np.array([ts.hour for ts in timestamps])
    days = np.array([ts.day for ts in timestamps])
    months = np.array([ts.month for ts in timestamps])
    weekdays = np.array([ts.weekday() for ts in timestamps])
    weekend_tx = ((weekdays == 5) | (weekdays == 6)).astype(int)
    night_tx = ((hours >= 23) | (hours <= 5)).astype(int)

    # 5. Transaction Types & Payment Methods
    tx_types = np.random.choice(TRANSACTION_TYPES, size=num_records, p=[0.45, 0.22, 0.15, 0.10, 0.08])
    pay_methods = np.random.choice(PAYMENT_METHODS, size=num_records, p=[0.35, 0.30, 0.15, 0.12, 0.08])
    auth_methods = np.random.choice(AUTH_METHODS, size=num_records, p=[0.38, 0.27, 0.18, 0.14, 0.03])
    dev_types = np.random.choice(DEVICE_TYPES, size=num_records, p=[0.42, 0.33, 0.15, 0.07, 0.03])

    # Assign merchants
    chosen_merch_indices = np.random.choice(NUM_MERCHANTS, size=num_records)
    chosen_merch_ids = [merchant_ids[i] for i in chosen_merch_indices]
    chosen_merch_cats = [merch_cats[i] for i in chosen_merch_indices]

    # 6. Customer Demographics mapped to transactions
    c_income = cust_df["customer_income"].values[chosen_cust_indices]
    c_balance = cust_df["account_balance"].values[chosen_cust_indices]
    c_age = cust_df["customer_age"].values[chosen_cust_indices]
    c_acc_age = cust_df["account_age_days"].values[chosen_cust_indices]
    c_home_loc = cust_df["customer_location"].values[chosen_cust_indices]
    c_prim_dev = cust_df["primary_device_id"].values[chosen_cust_indices]
    c_avg_amt = cust_df["average_transaction_amount"].values[chosen_cust_indices]
    c_freq = cust_df["transaction_frequency"].values[chosen_cust_indices]

    # Baseline transaction amounts centered around customer average with log-normal noise
    raw_amount = np.random.lognormal(mean=np.log(np.maximum(c_avg_amt, 15.0)), sigma=0.8, size=num_records)
    raw_amount = np.round(np.clip(raw_amount, 2.50, 25000.0), 2)

    # Devices & Locations
    is_new_device = (np.random.random(num_records) < 0.08).astype(int)
    txn_device_ids = []
    for i in range(num_records):
        if is_new_device[i] == 1:
            txn_device_ids.append(f"DEV_{np.random.randint(100000, 999999)}")
        else:
            txn_device_ids.append(c_prim_dev[i])

    is_new_location = (np.random.random(num_records) < 0.09).astype(int)
    txn_locations = []
    is_international = np.zeros(num_records, dtype=int)
    distances = np.zeros(num_records, dtype=float)

    for i in range(num_records):
        if is_new_location[i] == 1:
            loc = np.random.choice(LOCATIONS)
            txn_locations.append(loc)
            if loc not in LOCATIONS[:6]:
                is_international[i] = 1
                distances[i] = np.random.uniform(4000, 14000)
            else:
                distances[i] = np.random.uniform(80, 2500)
        else:
            txn_locations.append(c_home_loc[i])
            distances[i] = np.random.exponential(scale=12.0)

    distances = np.round(distances, 1)

    # IPs
    ip_addrs = [f"{np.random.randint(11, 215)}.{np.random.randint(10, 250)}.{np.random.randint(1, 254)}.{np.random.randint(1, 254)}" for _ in range(num_records)]

    # Velocity and failed attempt baselines
    failed_counts = np.random.choice([0, 1, 2, 3], size=num_records, p=[0.88, 0.09, 0.025, 0.005]).astype(int)
    login_attempts = (failed_counts + 1 + (np.random.random(num_records) < 0.06).astype(int)).astype(int)
    velocity_24h = (np.random.poisson(lam=2.5, size=num_records) + (raw_amount > 500).astype(int)).astype(float)

    # Previous amounts
    prev_amounts = np.round(np.roll(raw_amount, 1), 2)
    prev_amounts[0] = c_avg_amt[0]

    # Customer cumulative transaction count
    # Approximate realistic customer transaction count
    cust_tx_count = np.random.randint(10, 350, size=num_records)

    # 7. INJECT REALISTIC FRAUD SCENARIOS (~2.2% realistic imbalance)
    # Fraud rate target: ~2.0% - 2.5%
    fraud_labels = np.zeros(num_records, dtype=int)

    # Scenario A: Account Takeover (ATO)
    # High amount + new device + new location + failed logins + night hours
    ato_candidates = (
        (is_new_device == 1) &
        (is_new_location == 1) &
        (failed_counts >= 1) &
        (night_tx == 1) &
        (raw_amount > c_avg_amt * 2.5)
    )
    ato_indices = np.where(ato_candidates)[0]
    # Sample ~60% of these as confirmed fraud
    chosen_ato = np.random.choice(ato_indices, size=int(len(ato_indices) * 0.75), replace=False) if len(ato_indices) > 0 else []
    fraud_labels[chosen_ato] = 1

    # Scenario B: High-Value Balance Drain (Bust-Out / Wire Fraud)
    # Large wire transfer / cash_out draining > 80% of account balance
    drain_candidates = (
        np.isin(tx_types, ["TRANSFER", "CASH_OUT"]) &
        (raw_amount > 4000.0) &
        (raw_amount > c_balance * 0.70)
    )
    drain_indices = np.where(drain_candidates)[0]
    chosen_drain = np.random.choice(drain_indices, size=int(len(drain_indices) * 0.65), replace=False) if len(drain_indices) > 0 else []
    fraud_labels[chosen_drain] = 1

    # Scenario C: Card-Not-Present / Web Skimming on Luxury / Electronics
    cnp_candidates = (
        np.isin(chosen_merch_cats, ["Luxury Goods", "Electronics"]) &
        (pay_methods == "Credit Card") &
        (is_international == 1) &
        (distances > 3000)
    )
    cnp_indices = np.where(cnp_candidates)[0]
    chosen_cnp = np.random.choice(cnp_indices, size=int(len(cnp_indices) * 0.55), replace=False) if len(cnp_indices) > 0 else []
    fraud_labels[chosen_cnp] = 1

    # Scenario D: Velocity Burst Attacks
    velocity_candidates = (
        (velocity_24h >= 6) &
        (raw_amount > c_avg_amt * 3.0) &
        (auth_methods == "SMS OTP") # Intercepted OTP
    )
    vel_indices = np.where(velocity_candidates)[0]
    chosen_vel = np.random.choice(vel_indices, size=int(len(vel_indices) * 0.60), replace=False) if len(vel_indices) > 0 else []
    fraud_labels[chosen_vel] = 1

    # Scenario E: Inject targeted additional fraud to achieve exact realistic ~2.1% fraud rate
    current_fraud_count = fraud_labels.sum()
    target_fraud_count = int(num_records * 0.021) # 2.1%
    remaining_needed = target_fraud_count - current_fraud_count

    if remaining_needed > 0:
        # High risk candidate pool
        pool = np.where(
            (fraud_labels == 0) &
            ((raw_amount > 2500) | (is_new_device == 1) | (distances > 1500) | (failed_counts > 0))
        )[0]
        extra_fraud = np.random.choice(pool, size=min(remaining_needed, len(pool)), replace=False)
        fraud_labels[extra_fraud] = 1
        # Amplify fraud signals for these extra fraud cases realistically
        raw_amount[extra_fraud] = np.maximum(raw_amount[extra_fraud] * np.random.uniform(1.8, 3.5, size=len(extra_fraud)), 1200.0)
        failed_counts[extra_fraud] = np.random.choice([1, 2, 3], size=len(extra_fraud), p=[0.4, 0.4, 0.2])
        login_attempts[extra_fraud] = failed_counts[extra_fraud] + np.random.randint(1, 4, size=len(extra_fraud))
        velocity_24h[extra_fraud] += np.random.uniform(2.0, 5.0, size=len(extra_fraud))

    # Add realistic benign noise (legitimate users traveling or buying expensive items without fraud)
    benign_high = np.where((fraud_labels == 0) & (raw_amount > 5000))[0]
    logger.info("Benign high-value transactions count: %d", len(benign_high))

    # 8. Assemble Full DataFrame with all 35 required columns
    txn_ids = [f"TXN_{i:08d}" for i in range(1, num_records + 1)]
    cust_chosen_ids = [customer_ids[i] for i in chosen_cust_indices]
    ts_strings = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps]

    df = pd.DataFrame({
        "transaction_id": txn_ids,
        "customer_id": cust_chosen_ids,
        "transaction_timestamp": ts_strings,
        "transaction_amount": np.round(raw_amount, 2),
        "transaction_type": tx_types,
        "merchant_category": chosen_merch_cats,
        "merchant_id": chosen_merch_ids,
        "account_age_days": c_acc_age,
        "customer_age": c_age,
        "customer_income": c_income,
        "account_balance": c_balance,
        "transaction_location": txn_locations,
        "customer_location": c_home_loc,
        "device_id": txn_device_ids,
        "device_type": dev_types,
        "payment_method": pay_methods,
        "authentication_method": auth_methods,
        "ip_address": ip_addrs,
        "previous_transaction_amount": prev_amounts,
        "transaction_frequency": c_freq,
        "failed_transaction_count": failed_counts,
        "login_attempts": login_attempts,
        "international_transaction": is_international,
        "weekend_transaction": weekend_tx,
        "night_transaction": night_tx,
        "new_device": is_new_device,
        "new_location": is_new_location,
        "distance_from_home": distances,
        "transaction_velocity": np.round(velocity_24h, 1),
        "customer_transaction_count": cust_tx_count,
        "average_transaction_amount": np.round(c_avg_amt, 2),
        "transaction_hour": hours,
        "transaction_day": days,
        "transaction_month": months,
        "fraud_label": fraud_labels,
    })

    fraud_rate = df["fraud_label"].mean()
    logger.info("Dataset generation complete. Total: %d, Fraud: %d (%.3f%%)",
                len(df), df["fraud_label"].sum(), fraud_rate * 100)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved raw dataset to %s", output_path)

    if sample_path:
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        # Create a stratified 10,000 row sample for fast inspection
        sample_df = df.groupby("fraud_label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(10000 * len(x) / len(df))), random_state=seed)
        ).reset_index(drop=True)
        sample_df.to_csv(sample_path, index=False)
        logger.info("Saved sample dataset to %s (%d rows)", sample_path, len(sample_df))

    return df

if __name__ == "__main__":
    from src.config import RAW_DATA_FILE, SAMPLE_FILE
    generate_synthetic_bank_data(
        num_records=TOTAL_RECORDS,
        output_path=RAW_DATA_FILE,
        sample_path=SAMPLE_FILE,
    )
