from unittest.mock import patch, MagicMock
import pandas as pd
from src.modeling.inference import FraudScorer

# Change the patch target to just "joblib.load"
@patch("joblib.load") 
def test_inference_single_row(mock_load):
    # Setup Mock Model
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.8, 0.2]]
    mock_model.predict.return_value = [0]
    
    # Setup Mock Encoder
    mock_encoder = MagicMock()
    
    # Return model first, then encoder
    mock_load.side_effect = [mock_model, mock_encoder]

    scorer = FraudScorer(
        model_path="models/artifacts/fraud_model.joblib",
        encoders_path="models/encoders/preprocessing.joblib",
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

    # Force the return values to be standard types to avoid MagicMock formatting errors
    prob = scorer.predict_proba(df_input)
    label, action = scorer.predict_label_and_action(df_input)

    assert isinstance(prob, (float, int))
    assert 0.0 <= prob <= 1.0
    assert label in [0, 1]
    assert action in ["HARD_BLOCK", "OTP_CHALLENGE", "SOFT_REVIEW", "ALLOW"]