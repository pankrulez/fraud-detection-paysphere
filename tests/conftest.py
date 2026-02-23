import pandas as pd
import numpy as np
import pytest
import os

# generating a synthetic dataset for testing
@pytest.fixture
def sample_dataframe(n_rows: int = 5000, random_state: int = 42):
    """
    Generate synthetic transaction data matching transactions_fraud.csv schema.
    Default sample size = 5000 (same as production sampling).
    """
    np.random.seed(random_state)

    transaction_id = np.arange(1, n_rows + 1)
    customer_id = np.random.randint(100, 2000, n_rows)
    device_id = np.random.randint(10, 500, n_rows)
    merchant_id = np.random.randint(200, 800, n_rows)

    timestamps = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        np.random.randint(0, 30 * 24 * 60, n_rows), unit="m"
    )

    amount = np.round(np.random.exponential(scale=2000, size=n_rows), 2)

    payment_methods = np.random.choice(
        ["UPI", "CARD", "NETBANKING", "WALLET"], n_rows
    )

    is_international = np.random.choice([0, 1], n_rows, p=[0.9, 0.1])

    merchant_category = np.random.choice(
        ["Electronics", "Travel", "Grocery", "Fashion", "Food"], n_rows
    )

    ip_address_risk_score = np.round(np.random.uniform(0, 1, n_rows), 2)
    device_trust_score = np.round(np.random.uniform(0, 1, n_rows), 2)

    txn_count_last_24h = np.random.poisson(3, n_rows)
    avg_amount_last_24h = np.round(amount * np.random.uniform(0.8, 1.2, n_rows), 2)

    merchant_diversity_last_7d = np.random.randint(1, 6, n_rows)

    device_change_flag = np.random.choice([0, 1], n_rows, p=[0.85, 0.15])
    location_change_flag = np.random.choice([0, 1], n_rows, p=[0.85, 0.15])

    authentication_method = np.random.choice(["OTP", "PIN", "BIOMETRIC"], n_rows)

    otp_success_rate_customer = np.round(np.random.uniform(0.6, 1.0, n_rows), 2)

    past_fraud_count_customer = np.random.poisson(0.3, n_rows)
    past_disputes_customer = np.random.poisson(0.5, n_rows)

    merchant_historical_fraud_rate = np.round(np.random.uniform(0.01, 0.15, n_rows), 3)

    hour_of_day = timestamps.hour
    day_of_week = timestamps.dayofweek
    is_weekend = (day_of_week >= 5).astype(int)

    fraud_probability = (
        0.3 * is_international
        + 0.2 * (ip_address_risk_score > 0.7)
        + 0.2 * (device_trust_score < 0.3)
        + 0.2 * (past_fraud_count_customer > 0)
        + 0.1 * (amount > 5000)
    )

    is_fraud = (np.random.rand(n_rows) < fraud_probability).astype(int)

    df = pd.DataFrame({
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "device_id": device_id,
        "merchant_id": merchant_id,
        "timestamp": timestamps,
        "amount": amount,
        "payment_method": payment_methods,
        "is_international": is_international,
        "merchant_category": merchant_category,
        "ip_address_risk_score": ip_address_risk_score,
        "device_trust_score": device_trust_score,
        "txn_count_last_24h": txn_count_last_24h,
        "avg_amount_last_24h": avg_amount_last_24h,
        "merchant_diversity_last_7d": merchant_diversity_last_7d,
        "device_change_flag": device_change_flag,
        "location_change_flag": location_change_flag,
        "authentication_method": authentication_method,
        "otp_success_rate_customer": otp_success_rate_customer,
        "past_fraud_count_customer": past_fraud_count_customer,
        "past_disputes_customer": past_disputes_customer,
        "merchant_historical_fraud_rate": merchant_historical_fraud_rate,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_fraud": is_fraud,
    })

    return df

@pytest.fixture
def setup_test_data(sample_dataframe):
    """Creates the local directory and CSV file needed for tests."""
    # Define the path expected by the test
    raw_dir = "data/raw"
    raw_path = os.path.join(raw_dir, "transactions_fraud.csv")
    
    # Create directory if it doesn't exist
    os.makedirs(raw_dir, exist_ok=True)
    
    # Save the mock dataframe to that path
    sample_dataframe.to_csv(raw_path, index=False)
    
    yield raw_path # Provide the path to the test
    
    # Optional: Clean up after the test is done
    if os.path.exists(raw_path):
        os.remove(raw_path)