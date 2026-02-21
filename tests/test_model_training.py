import os
import pandas as pd
from src.modeling.train import train_pipeline
from src.utils.io_utils import load_model

def test_train_pipeline(tmp_path, monkeypatch, sample_dataframe):
    # 1. Physically create the missing directory and file
    raw_dir = "data/raw"
    raw_file = os.path.join(raw_dir, "transactions_fraud.csv")
    os.makedirs(raw_dir, exist_ok=True)
    sample_dataframe.to_csv(raw_file, index=False)

    # 2. Override config paths to use tmp dirs
    def fake_config():
        return {
            "data": {
                "raw_path": raw_file, # Now this file actually exists!
                "interim_path": str(tmp_path / "interim.csv"),
                "processed_path": str(tmp_path / "processed.csv"),
            },
            "model": {
                "target_column": "is_fraud",
                "test_size": 0.2,
                "random_state": 42,
                "algorithm": "random_forest",
            },
            "threshold": {
                "fraud_cutoff": 0.5
            },
        }

    # Monkeypatch config loader in the training module
    from src.modeling import train as train_module
    monkeypatch.setattr(train_module, "load_config", lambda path: fake_config())

    # 3. Run the pipeline (this creates the .joblib file)
    train_pipeline()

    # 4. Verify the model was saved and can be loaded
    # Ensure this path matches where train_pipeline saves the model
    model = load_model("models/artifacts/fraud_model.joblib")
    assert model is not None