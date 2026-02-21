import os
import pandas as pd
from src.modeling.train import train_pipeline
from src.utils.io_utils import load_model

def test_train_pipeline(tmp_path, monkeypatch, sample_dataframe):
    # 1. Filter fraud rows and duplicate them to ensure >= 2 members
    fraud_rows = sample_dataframe[sample_dataframe['is_fraud'] == 1]
    
    # Concatenate original data with enough fraud copies to satisfy stratification
    # With test_size=0.2, you need at least 5-10 rows and 2+ fraud cases
    balanced_df = pd.concat([sample_dataframe] + [fraud_rows] * 10, ignore_index=True)
    
    # 2. Setup the directory and file as before
    raw_dir = "data/raw"
    raw_file = os.path.join(raw_dir, "transactions_fraud.csv")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs("models/artifacts", exist_ok=True) 
    balanced_df.to_csv(raw_file, index=False)

    # 3. Rest of your config and monkeypatching...
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

    # 4. Run the pipeline
    train_pipeline()
    
    model = load_model("models/artifacts/fraud_model.joblib")
    assert model is not None