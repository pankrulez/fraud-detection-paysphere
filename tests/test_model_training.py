import os
import pandas as pd
from unittest.mock import patch
from src.modeling.train import train_pipeline

def test_train_pipeline(tmp_path, monkeypatch, sample_dataframe):
    # 1. Create a truly balanced dataset (6 rows: 3 fraud, 3 non-fraud)
    fraud_row = sample_dataframe[sample_dataframe['is_fraud'] == 1]
    clean_rows = sample_dataframe[sample_dataframe['is_fraud'] == 0]
    
    # Ensure we have at least 3 of each to satisfy StratifiedShuffleSplit
    balanced_df = pd.concat([fraud_row]*3 + [clean_rows]*3, ignore_index=True)
    
    # 2. Setup physical paths for the runner
    raw_file = "data/raw/transactions_fraud.csv"
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("models/artifacts", exist_ok=True)
    balanced_df.to_csv(raw_file, index=False)

    # 3. Double-layer Mocking: 
    # Layer A: Mock the config
    def fake_config(path=None):
        return {
            "data": {
                "raw_path": raw_file,
                "interim_path": str(tmp_path / "interim.csv"),
                "processed_path": str(tmp_path / "processed.csv"),
            },
            "model": {
                "target_column": "is_fraud",
                "test_size": 0.2,
                "random_state": 42,
                "algorithm": "random_forest",
            },
        }

    # Layer B: Mock the data loader to return our balanced_df directly
    # Adjust 'src.modeling.train.pd.read_csv' to wherever your code loads data
    with patch('src.modeling.train.pd.read_csv', return_value=balanced_df):
        from src.modeling import train as train_module
        monkeypatch.setattr(train_module, "load_config", fake_config)
        
        # 4. Run the pipeline
        train_pipeline()

    # 5. Verify output
    assert os.path.exists("models/artifacts/fraud_model.joblib")