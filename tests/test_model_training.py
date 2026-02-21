import os
import pandas as pd
from src.modeling.train import train_pipeline
from src.utils.io_utils import load_model

def test_train_pipeline(tmp_path, monkeypatch, sample_dataframe):
    # 1. Double the dataframe rows to ensure at least 2 fraud cases
    balanced_df = pd.concat([sample_dataframe, sample_dataframe], ignore_index=True)
    
    # 2. Create the missing directory and file
    raw_dir = "data/raw"
    raw_file = os.path.join(raw_dir, "transactions_fraud.csv")
    os.makedirs(raw_dir, exist_ok=True)
    balanced_df.to_csv(raw_file, index=False)

    # 3. Rest of your config override code...
    def fake_config():
        return {
            "data": {
                "raw_path": raw_file,
                "interim_path": str(tmp_path / "interim.csv"),
                "processed_path": str(tmp_path / "processed.csv"),
            },
            "model": {
                "target_column": "is_fraud",
                "test_size": 0.2, # With 6 rows, 20% is ~1 row for testing
                "random_state": 42,
                "algorithm": "random_forest",
            },
            "threshold": {"fraud_cutoff": 0.5},
        }

    from src.modeling import train as train_module
    monkeypatch.setattr(train_module, "load_config", lambda path: fake_config())

    # 4. Run pipeline
    train_pipeline()
    
    # Ensure this path matches your train_pipeline's output
    # You might need to os.makedirs("models/artifacts", exist_ok=True) if the code doesn't
    model = load_model("models/artifacts/fraud_model.joblib")
    assert model is not None