import pandas as pd
from src.features.feature_engineering import engineer_behavioral_features, prepare_features


def test_engineer_behavioral_features(sample_dataframe):
    df_feat = engineer_behavioral_features(sample_dataframe.copy())
    assert "amount_deviation" in df_feat.columns
    assert "combined_risk_index" in df_feat.columns


def test_prepare_features(sample_dataframe):
    df_feat, y, encoders = prepare_features(sample_dataframe.copy(), target_col="is_fraud")
    assert df_feat.shape[0] == len(y)
    assert "ohe" in encoders and "scaler" in encoders