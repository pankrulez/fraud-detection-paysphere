import pandas as pd
from typing import List
from src.logger import get_logger
from src.exceptions import DataSchemaError, BusinessRuleViolationError

logger = get_logger(__name__)


EXPECTED_COLUMNS = [
    "transaction_id",
    "customer_id",
    "device_id",
    "merchant_id",
    "timestamp",
    "amount",
    "payment_method",
    "is_international",
    "merchant_category",
    "ip_address_risk_score",
    "device_trust_score",
    "txn_count_last_24h",
    "avg_amount_last_24h",
    "merchant_diversity_last_7d",
    "device_change_flag",
    "location_change_flag",
    "authentication_method",
    "otp_success_rate_customer",
    "past_fraud_count_customer",
    "past_disputes_customer",
    "merchant_historical_fraud_rate",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_fraud",
]


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates that all expected columns are present and types are reasonable.

    Raises DataSchemaError if validation fails.
    """
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        logger.error(f"Missing columns: {missing}")
        raise DataSchemaError(f"Missing columns: {missing}")

    # Example: ensure numeric columns are numeric
    numeric_cols = [
        "amount",
        "ip_address_risk_score",
        "device_trust_score",
        "txn_count_last_24h",
        "avg_amount_last_24h",
        "merchant_diversity_last_7d",
        "device_change_flag",
        "location_change_flag",
        "otp_success_rate_customer",
        "past_fraud_count_customer",
        "past_disputes_customer",
        "merchant_historical_fraud_rate",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_fraud",
    ]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.info(f"Casting {col} to numeric")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def validate_business_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply key business rules from the data definition and raise errors or log warnings when violated.
    """
    # Example rules drawn from the dictionary:
    # - amount must be > 0
    if (df["amount"] <= 0).any():
        invalid_rows = df[df["amount"] <= 0]
        logger.warning(f"Found non-positive amounts in {len(invalid_rows)} rows.")
        raise BusinessRuleViolationError("Non-positive amounts found.")

    # - time_of_day must be between 0 and 23
    if df["hour_of_day"].min() < 0 or df["hour_of_day"].max() > 23:
        logger.warning("time_of_day out of 0–23 range.")
        raise BusinessRuleViolationError("Invalid time_of_day value.")

    # - day_of_week must be between 0 and 6
    if df["day_of_week"].min() < 0 or df["day_of_week"].max() > 6:
        logger.warning("day_of_week out of 0–6 range.")
        raise BusinessRuleViolationError("Invalid day_of_week value.")

    # - risk scores between 0 and 1
    risk_cols = [
        "ip_address_risk_score",
        "device_trust_score",
        "merchant_historical_fraud_rate",
        "otp_success_rate_customer",
    ]
    for col in risk_cols:
        if df[col].min() < 0 or df[col].max() > 1:
            logger.warning(f"{col} must be in [0,1].")
            raise BusinessRuleViolationError(f"Invalid risk score in {col}.")

    return df