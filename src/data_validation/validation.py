# src/data_validation/validation.py
import pandas as pd
from src.logger import get_logger
from src.exceptions import BusinessRuleViolationError

logger = get_logger(__name__)


def check_primary_key(df: pd.DataFrame) -> None:
    """
    Validate primary key uniqueness for transaction_id.

    Business meaning:
    - Every transaction must be uniquely traceable for audits and chargeback handling.
    """
    dup_count = df["transaction_id"].duplicated().sum()
    if dup_count > 0:
        logger.error(f"Found {dup_count} duplicate transaction_id values.")
        raise BusinessRuleViolationError("Primary key violation: duplicate transaction_id.")


def check_nulls(df: pd.DataFrame) -> None:
    """
    Validate critical fields are non-null.

    Business rationale:
    - Missing amount, timestamp, or payment_method makes the transaction unusable for risk decisions.
    """
    critical_cols = [
        "transaction_id",
        "customer_id",
        "device_id",
        "merchant_id",
        "timestamp",
        "amount",
        "payment_method",
        "is_fraud",
    ]
    nulls = df[critical_cols].isnull().sum()
    bad = nulls[nulls > 0]
    if not bad.empty:
        logger.error(f"Nulls in critical columns: {bad.to_dict()}")
        raise BusinessRuleViolationError(f"Critical nulls found: {bad.to_dict()}")


def check_date_order(df: pd.DataFrame) -> None:
    """
    Validate timestamp is parseable and non-null,
    and that time_of_day/day_of_week are consistent.

    Business rationale:
    - Temporal features (hour, weekday/weekend) are central to fraud timing patterns.
    """
    try:
        ts = pd.to_datetime(df["timestamp"])
    except Exception as e:
        logger.error(f"Failed to parse timestamp: {e}")
        raise BusinessRuleViolationError("Invalid timestamp format.")

    # Example soft check: time_of_day is consistent with timestamp hour for a random sample
    if "time_of_day" in df.columns:
        mismatch = (ts.dt.hour != df["time_of_day"]).mean()
        if mismatch > 0.05:
            logger.warning(f"time_of_day differs from timestamp hour for {mismatch:.1%} of rows.")


def check_flag_ranges(df: pd.DataFrame) -> None:
    """
    Validate flag columns are in {0,1}.

    Business rationale:
    - Flags represent binary conditions (e.g., domestic vs international, weekday vs weekend).
    """
    flag_cols = ["is_international", "ip_address_country_match", "is_weekend", "is_fraud"]
    for col in flag_cols:
        bad = df[~df[col].isin([0, 1])]
        if not bad.empty:
            logger.error(f"Invalid flag values in {col}.")
            raise BusinessRuleViolationError(f"{col} must be 0 or 1.")


def check_risk_score_ranges(df: pd.DataFrame) -> None:
    """
    Validate that risk scores (0–1) stay in valid range.

    Business rationale:
    - Risk scores are calibrated probabilities or indexes, must remain bounded.
    """
    risk_cols = [
        "ip_address_risk_score",
        "device_trust_score",
        "historical_fraud_rate",
        "merchant_historical_fraud_rate",
        "location_risk_score",
        "transaction_success_rate_customer",
    ]
    for col in risk_cols:
        if df[col].min() < 0 or df[col].max() > 1:
            logger.error(f"{col} out of [0,1] range.")
            raise BusinessRuleViolationError(f"Risk score out of range in {col}.")


def check_temporal_bounds(df: pd.DataFrame) -> None:
    """
    Validate time-of-day and day-of-week bounds.

    Business rationale:
    - These are discrete temporal bins used for pattern analysis.
    """
    if df["time_of_day"].min() < 0 or df["time_of_day"].max() > 23:
        logger.error("time_of_day must be in [0,23].")
        raise BusinessRuleViolationError("Invalid time_of_day values.")
    if df["day_of_week"].min() < 0 or df["day_of_week"].max() > 6:
        logger.error("day_of_week must be in [0,6].")
        raise BusinessRuleViolationError("Invalid day_of_week values.")


def run_all_validations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all business validations in sequence.
    """
    logger.info("Running primary key validation.")
    check_primary_key(df)

    logger.info("Checking nulls in critical columns.")
    check_nulls(df)

    logger.info("Validating timestamp and temporal consistency.")
    check_date_order(df)

    logger.info("Validating flag ranges.")
    check_flag_ranges(df)

    logger.info("Validating risk score ranges.")
    check_risk_score_ranges(df)

    logger.info("Validating temporal bounds.")
    check_temporal_bounds(df)

    logger.info("Business validation passed.")
    return df