import os
import pandas as pd
from src.modeling.train import train_pipeline
from src.utils.io_utils import load_model

def test_train_pipeline(tmp_path, monkeypatch, sample_dataframe):
    # Ensure we have at least 4 fraud cases to satisfy the 20% split
    fraud_rows = sample_dataframe[sample_dataframe['is_fraud'] == 1]
    # Multiply the fraud rows specifically
    balanced_df = pd.concat([sample_dataframe] + [fraud_rows] * 5, ignore_index=True)
    
    raw_dir = "data/raw"
    raw_file = os.path.join(raw_dir, "transactions_fraud.csv")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs("models/artifacts", exist_ok=True) # Ensure model dir exists too
    balanced_df.to_csv(raw_file, index=False)

    def fake_config():
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
            "threshold": {"fraud_cutoff": 0.5},
        }

    from src.modeling import train as train_module
    monkeypatch.setattr(train_module, "load_config", lambda path: fake_config())

    train_pipeline()
    
    model = load_model("models/artifacts/fraud_model.joblib")
    assert model is not None