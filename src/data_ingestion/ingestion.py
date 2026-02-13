import pandas as pd
from src.logger import get_logger
from src.utils.io_utils import read_csv, write_csv
from src.utils.validation_utils import validate_schema, validate_business_rules

logger = get_logger(__name__)


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Loads raw transaction data from CSV.
    """
    df = read_csv(path)
    logger.info(f"Loaded raw data with shape: {df.shape}")
    return df


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates schema and business rules, handles basic cleaning such as dropping duplicates.
    """
    df = validate_schema(df)
    df = validate_business_rules(df)

    # Drop duplicate transaction_id if any
    before = df.shape[0]
    df = df.drop_duplicates(subset=["transaction_id"])
    after = df.shape[0]
    if before != after:
        logger.info(f"Dropped {before - after} duplicate transactions.")

    # Handle missing values: basic strategy (can be refined)
    df = df.dropna(subset=["is_fraud"])
    df = df.fillna({
        "velocity_1h": 0,
        "velocity_24h": 0,
        "velocity_7d": 0,
        "customer_tenure_days": 0,
        "transaction_success_rate_customer": 1.0,
    })

    return df


def ingest_and_validate(raw_path: str, interim_path: str) -> pd.DataFrame:
    """
    Full ingestion routine: load, validate, clean, and save interim data.
    """
    df = load_raw_data(raw_path)
    df = validate_and_clean(df)
    write_csv(df, interim_path)
    logger.info(f"Interim data saved to {interim_path}")
    return df