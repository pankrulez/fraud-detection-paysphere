from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.modeling.inference import FraudScorer

@patch("joblib.load") 
def test_inference_single_row(mock_load):
    # 1. Setup Mock Unified Pipeline
    mock_pipeline = MagicMock()
    # Mock the internal sklearn pipeline's predict_proba
    mock_pipeline.predict_proba.return_value = np.array([[0.8, 0.2]]) 
    
    # joblib.load returns the pipeline
    mock_load.return_value = mock_pipeline

    # 2. Initialize Scorer 
    # Pass the path as a positional argument to avoid "unexpected keyword" errors
    scorer = FraudScorer("models/artifacts/fraud_pipeline.joblib")

    # 3. Build valid input matching your 22-column schema
    df_input = pd.DataFrame([{
        "transaction_id": "T0", "customer_id": "C0", "device_id": "D0", "merchant_id": "M0",
        "timestamp": "2026-01-01 00:00:00", "amount": 1000.0, "payment_method": "UPI",
        "is_international": 0, "merchant_category": "Electronics", "ip_address_risk_score": 0.2,
        "device_trust_score": 0.8, "txn_count_last_24h": 3, "location_change_flag": 0,
        "otp_success_rate_customer": 0.9, "past_fraud_count_customer": 0, "past_disputes_customer": 0,
        "merchant_historical_fraud_rate": 0.05, "hour_of_day": 12, "is_weekend": 0,
        "ip_address_country_match": 1, "customer_tenure_days": 365, "day_of_week": 2
    }])

    # 4. Execute and Unpack correctly
    # Note: predict_label_and_action now returns (label, action, prob)
    prob = scorer.predict_proba(df_input)
    label, action, score = scorer.predict_label_and_action(df_input)

    # 5. Assertions
    assert isinstance(prob, (float, np.floating))
    assert 0.0 <= prob <= 1.0
    assert label in [0, 1]
    assert action in ["HARD_BLOCK", "MANUAL_REVIEW", "OTP_VERIFICATION", "ALLOW"]