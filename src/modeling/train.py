import yaml
import optuna
import pandas as pd
import numpy as np
import datetime
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier

from src.logger import get_logger
from src.utils.io_utils import write_csv, save_model
from src.utils.metrics_utils import compute_classification_metrics
from src.data_ingestion.ingestion import ingest_and_validate
from src.features.feature_engineering import engineer_behavioral_features, build_preprocessing_pipeline

logger = get_logger("fraud")
version = datetime.datetime.now().strftime("%Y%m%d-%H%M")

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def drop_identifiers(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Separates features and target, dropping IDs that shouldn't be modeled."""
    y = df[target_col]
    id_cols = ["transaction_id", "customer_id", "device_id", "merchant_id", "timestamp"]
    X = df.drop(columns=[target_col] + id_cols, errors="ignore")
    return X, y

def optimize_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series, base_pipeline) -> dict:
    """Uses Optuna to find the best hyperparameters via Time-Series Cross Validation."""
    logger.info("Starting Optuna hyperparameter optimization...")

    def objective(trial):
        # Define hyperparameter search space (Assuming RandomForest for this example)
        rf_max_depth = trial.suggest_int("max_depth", 5, 20)
        rf_n_estimators = trial.suggest_int("n_estimators", 50, 300)
        rf_min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

        # Use TimeSeriesSplit to prevent leakage during cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        pr_auc_scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Clone the base pipeline and append the classifier
            pipeline = clone(base_pipeline)
            classifier = RandomForestClassifier(
                max_depth=rf_max_depth,
                n_estimators=rf_n_estimators,
                min_samples_leaf=rf_min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
            pipeline.steps.append(('classifier', classifier))

            # Train and evaluate
            pipeline.fit(X_fold_train, y_fold_train)
            preds = pipeline.predict_proba(X_fold_val)[:, 1]
            score = average_precision_score(y_fold_val, preds)
            pr_auc_scores.append(score)

        return float(np.mean(pr_auc_scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10) # Set to 10 for speed, increase to 50+ later
    
    logger.info(f"Best Optuna PR-AUC Score: {study.best_value:.4f}")
    logger.info(f"Best Params: {study.best_params}")
    return study.best_params

def train_pipeline(config_path: str = "config/config.yaml") -> None:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    threshold_cfg = cfg["threshold"]

    # 1. Ingest Data
    raw_df = ingest_and_validate(raw_path=data_cfg["raw_path"], interim_path=data_cfg["interim_path"])

    # 2. Engineer Features (Leakage-Proof)
    df = engineer_behavioral_features(raw_df)
    
    # 3. Chronological Train/Test Split (Out-of-Time Validation)
    logger.info("Performing chronological train/test split...")
    df = df.sort_values("timestamp")
    split_idx = int(len(df) * (1 - model_cfg.get("test_size", 0.2)))
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train, y_train = drop_identifiers(train_df, model_cfg["target_column"])
    X_test, y_test = drop_identifiers(test_df, model_cfg["target_column"])

    # 4. Build Preprocessing Pipeline (Scaling + OHE + SMOTE)
    preprocessor_pipeline = build_preprocessing_pipeline()

    # 5. Hyperparameter Tuning
    best_params = optimize_hyperparameters(X_train, y_train, preprocessor_pipeline)

    # 6. Train Final Model on Full Training Set
    logger.info("Training final pipeline with best parameters...")
    final_pipeline = clone(preprocessor_pipeline)
    final_classifier = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    final_pipeline.steps.append(('classifier', final_classifier))
    
    final_pipeline.fit(X_train, y_train)
    
    logger.info(f"V-{version} model training completed.")

    # 7. Evaluate on Out-of-Time Test Set
    y_prob = final_pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_classification_metrics(y_true=y_test, y_prob=y_prob, threshold=threshold_cfg["fraud_cutoff"])
    logger.info(f"OOT Test Metrics at threshold {threshold_cfg['fraud_cutoff']}: {metrics}")

    # 8. Save Test Set Predictions
    test_out = test_df.copy()
    test_out["pred_prob"] = y_prob
    write_csv(test_out, data_cfg["processed_path"])

    # 9. Save Unified Artifact (Pipeline + Model all in one)
    save_model(final_pipeline, "models/artifacts/fraud_pipeline.joblib")
    logger.info("Unified fraud pipeline saved successfully.")

if __name__ == "__main__":
    train_pipeline()