from src.modeling.train import train_pipeline
from src.utils.io_utils import load_model


def test_train_pipeline(tmp_path, monkeypatch):
    # Override config paths to use tmp dirs
    def fake_config():
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
            "threshold": {
                "fraud_cutoff": 0.5
            },
        }

    from src import modeling
    # Monkeypatch config loader
    from src.modeling import train as train_module
    monkeypatch.setattr(train_module, "load_config", lambda path: fake_config())

    train_pipeline()
    model = load_model("models/artifacts/fraud_model.joblib")
    assert model is not None