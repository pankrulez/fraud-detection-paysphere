import pandas as pd
import pytest
import os


@pytest.fixture
def sample_dataframe():
    """
    Small sample of the real schema, matching transactions_fraud.csv columns.

    Only a few rows are needed; values are arbitrary but realistic.
    """
    data = {
        "transaction_id": [1, 2, 3],
        "customer_id": [100, 100, 101],
        "device_id": [10, 10, 11],
        "merchant_id": [200, 201, 200],
        "timestamp": ["2024-01-01 10:00:00", "2024-01-01 11:00:00", "2024-01-02 09:30:00"],
        "amount": [1000.0, 1500.0, 500.0],
        "payment_method": ["UPI", "CARD", "NETBANKING"],
        "is_international": [0, 1, 0],
        "merchant_category": ["Electronics", "Travel", "Grocery"],
        "ip_address_risk_score": [0.1, 0.5, 0.2],
        "device_trust_score": [0.9, 0.4, 0.8],
        "txn_count_last_24h": [2, 5, 1],
        "avg_amount_last_24h": [900.0, 1400.0, 600.0],
        "merchant_diversity_last_7d": [1, 3, 2],
        "device_change_flag": [0, 1, 0],
        "location_change_flag": [0, 1, 0],
        "authentication_method": ["OTP", "PIN", "OTP"],
        "otp_success_rate_customer": [0.95, 0.7, 0.98],
        "past_fraud_count_customer": [0, 1, 0],
        "past_disputes_customer": [0, 2, 0],
        "merchant_historical_fraud_rate": [0.02, 0.1, 0.03],
        "hour_of_day": [10, 11, 9],
        "day_of_week": [0, 0, 1],
        "is_weekend": [0, 0, 0],
        "is_fraud": [0, 1, 0],
    }
    df = pd.DataFrame(data)
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