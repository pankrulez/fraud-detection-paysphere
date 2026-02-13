import os
import joblib
import pandas as pd
from typing import Any
from src.logger import get_logger

logger = get_logger(__name__)


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def read_csv(path: str) -> pd.DataFrame:
    logger.info(f"Reading CSV from {path}")
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    logger.info(f"Writing CSV to {path}, shape={df.shape}")
    df.to_csv(path, index=False)


def save_model(model: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    logger.info(f"Saving model to {path}")
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    logger.info(f"Loading model from {path}")
    return joblib.load(path)