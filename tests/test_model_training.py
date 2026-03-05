import os
import pandas as pd
from unittest.mock import patch
from src.modeling.train import train_pipeline

def test_train_pipeline(tmp_path, monkeypatch, sample_dataframe):
    # 1. Create a balanced dataset
    fraud_rows = sample_dataframe[sample_dataframe['is_fraud'] == 1]
    clean_rows = sample_dataframe[sample_dataframe['is_fraud'] == 0]
    balanced_df = pd.concat([fraud_rows] * 10 + [clean_rows] * 10, ignore_index=True)

    # 2. Mock the config
    def fake_config(path=None):
        return {
            "data": {
                "raw_path": "data/raw/transactions_fraud.csv",
                "interim_path": str(tmp_path / "interim.csv"),
                "processed_path": str(tmp_path / "processed.csv"),
            },
            "model": {
                "target_column": "is_fraud",
                "test_size": 0.2,
                "random_state": 42,
                "algorithm": "random_forest",
            },
            "threshold": {"fraud_cutoff": 0.5},
        }

    from src.modeling import train as train_module
    
    os.makedirs("models/artifacts", exist_ok=True)
    # We don't strictly need models/encoders anymore, but it doesn't hurt

    monkeypatch.setattr(train_module, "load_config", fake_config)
    
    with patch("src.modeling.train.ingest_and_validate", return_value=balanced_df):
        train_pipeline()

    # 5. Verify the UNIFIED artifact was created
    assert os.path.exists("models/artifacts/fraud_pipeline.joblib")