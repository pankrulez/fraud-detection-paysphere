import yaml
from sklearn.model_selection import train_test_split

from src.logger import get_logger
from src.utils.io_utils import write_csv, save_model
from src.utils.metrics_utils import compute_classification_metrics
from src.data_ingestion.ingestion import ingest_and_validate
from src.features.feature_engineering import prepare_features, handle_imbalance
from src.modeling.model_definition import get_model

logger = get_logger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_pipeline(config_path: str = "config/config.yaml") -> None:
    """
    End-to-end training pipeline:
    - Ingests and validates data.
    - Engineers features.
    - Handles imbalance.
    - Trains model and evaluates precision/recall for fraud.
    - Saves model and encoders.
    """
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    threshold_cfg = cfg["threshold"]

    df = ingest_and_validate(
        raw_path=data_cfg["raw_path"],
        interim_path=data_cfg["interim_path"],
    )

    X, y, encoders = prepare_features(df, target_col=model_cfg["target_column"], fit=True)

    # Train/test split (stratified to preserve fraud ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=model_cfg["test_size"], random_state=model_cfg["random_state"], stratify=y
    )

    # Handle class imbalance on training set only
    X_train_res, y_train_res = handle_imbalance(X_train, y_train)

    model = get_model(model_cfg["algorithm"])
    logger.info("Training model...")
    model.fit(X_train_res, y_train_res)
    logger.info("Model training completed.")

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_classification_metrics(
        y_true=y_test,
        y_prob=y_prob,
        threshold=threshold_cfg["fraud_cutoff"],
    )
    logger.info(f"Evaluation metrics at threshold {threshold_cfg['fraud_cutoff']}: {metrics}")

    # Save processed features for any additional analysis (optional)
    # write_csv(
    #    df.assign(pred_prob=y_prob),
    #    data_cfg["processed_path"],
    # )
    
    # save just metrics or test-set predictions if needed
    # For now, skip writing preds onto the full df to avoid length mismatch
    test_out = X_test.copy()
    test_out["is_fraud"] = y_test.values
    test_out["pred_prob"] = y_prob
    write_csv(test_out, data_cfg["processed_path"])
    

    # Save artifacts
    save_model(model, "models/artifacts/fraud_model.joblib")
    save_model(encoders, "models/encoders/preprocessing.joblib")
    logger.info("Model and encoders saved.")


if __name__ == "__main__":
    train_pipeline()