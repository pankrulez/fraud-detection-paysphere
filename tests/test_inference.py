import pandas as pd
from unittest.mock import patch, MagicMock
from src.modeling.inference import FraudScorer

@patch("joblib.load") # This intercepts joblib.load inside FraudScorer
def test_inference_single_row(mock_load):
    # 1. Setup Mock Objects
    # We simulate a model that has a predict_proba method
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.1, 0.9]] # Simulates 90% fraud probability
    
    # We simulate encoders (can just be a dummy object)
    mock_encoder = MagicMock()

    # Define what joblib.load returns each time it's called
    # First call (model_path) -> mock_model
    # Second call (encoders_path) -> mock_encoder
    mock_load.side_effects = [mock_model, mock_encoder]

    # 2. Initialize Scorer (it won't crash now because joblib.load is mocked)
    scorer = FraudScorer(
        model_path="models/artifacts/fraud_model.joblib",
        encoders_path="models/encoders/preprocessing.joblib",
        threshold=0.5,
    )

    df_input = pd.DataFrame([{
        "transaction_id": 0, "customer_id": 0, "device_id": 0, "merchant_id": 0,
        "timestamp": "2024-01-01 00:00:00", "amount": 1000.0, "payment_method": "UPI",
        "is_international": 0, "merchant_category": "Electronics", "ip_address_risk_score": 0.2,
        "device_trust_score": 0.8, "velocity_1h": 1, "velocity_24h": 3, "velocity_7d": 10,
        "customer_tenure_days": 200, "historical_fraud_rate": 0.0, "merchant_historical_fraud_rate": 0.05,
        "ip_address_country_match": 1, "previous_chargeback_count": 0, "time_of_day": 12,
        "day_of_week": 2, "is_weekend": 0, "location_risk_score": 0.1,
        "transaction_success_rate_customer": 0.98, "is_fraud": 0,
    }])

    # 3. Run Inference
    prob = scorer.predict_proba(df_input)
    label, action = scorer.predict_label_and_action(df_input)

    # 4. Assertions
    assert 0.0 <= prob <= 1.0
    assert label in [0, 1]
    assert action in ["HARD_BLOCK", "OTP_CHALLENGE", "SOFT_REVIEW", "ALLOW"]