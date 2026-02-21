import os
import pandas as pd
from unittest.mock import patch
from src.modeling.train import train_pipeline

def test_train_pipeline(tmp_path, monkeypatch, sample_dataframe):
    # 1. Create a balanced dataset (at least 10 rows, 50% fraud)
    # This ensures StratifiedShuffleSplit always has enough samples
    fraud_rows = sample_dataframe[sample_dataframe['is_fraud'] == 1]
    clean_rows = sample_dataframe[sample_dataframe['is_fraud'] == 0]
    balanced_df = pd.concat([fraud_rows] * 10 + [clean_rows] * 10, ignore_index=True)

    # 2. Mock the config to use tmp_path for outputs
    def fake_config(path=None):
        return {
            "data": {
                "raw_path": "data/raw/transactions_fraud.csv", # Placeholder
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

    # 3. Patch BOTH the config loader and the ingestion function
    from src.modeling import train as train_module
    
    # Ensure directories exist so save_model doesn't fail
    os.makedirs("models/artifacts", exist_ok=True)
    os.makedirs("models/encoders", exist_ok=True)

    monkeypatch.setattr(train_module, "load_config", fake_config)
    
    # We patch 'ingest_and_validate' where it is USED (in train_module)
    with patch("src.modeling.train.ingest_and_validate", return_value=balanced_df):
        # 4. Run the pipeline
        train_pipeline()

    # 5. Verify the joblib files were actually created
    assert os.path.exists("models/artifacts/fraud_model.joblib")
    assert os.path.exists("models/encoders/preprocessing.joblib")