from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib

from src.logger import get_logger

logger = get_logger(__name__)

# --- Column Definitions ---
CATEGORICAL_COLS = ["payment_method", "merchant_category"]
BINARY_COLS = ["is_international", "ip_address_country_match", "is_weekend"]

BASE_NUMERIC_COLS = [
    "amount", "ip_address_risk_score", "device_trust_score",
    "velocity_1h", "velocity_24h", "velocity_7d", "customer_tenure_days",
    "historical_fraud_rate", "merchant_historical_fraud_rate",
    "previous_chargeback_count", "time_of_day", "day_of_week",
    "location_risk_score", "transaction_success_rate_customer"
]

# Features created inside engineer_behavioral_features
ENGINEERED_NUMERIC_COLS = [
    "amount_deviation", "velocity_ratio_1h_24h", "velocity_ratio_24h_7d",
    "device_customer_sharing", "combined_risk_index"
]

# combine all numeric and binary columns to scale them together
ALL_NUMERIC_COLS = BASE_NUMERIC_COLS + BINARY_COLS + ENGINEERED_NUMERIC_COLS


def engineer_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Leakage-proof feature engineering using cumulative historical windows.
    """
    df = df.copy()
    
    # 1. Ensure data is sorted by time to prevent future-peeking
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
    else:
        logger.warning("No timestamp column found! Assuming data is chronologically ordered.")

    # --- Base Features (No Leakage Risk) ---
    if "historical_fraud_rate" not in df.columns:
        fraud = df.get("past_fraud_count_customer", 0)
        disputes = df.get("past_disputes_customer", 0)
        df["historical_fraud_rate"] = (fraud + disputes) / (fraud + disputes + 1.0)
        
    if "txn_count_last_24h" in df.columns:
        df["velocity_24h"] = df["txn_count_last_24h"]
        df["velocity_1h"] = (df["txn_count_last_24h"] / 6).round().astype(int)
        df["velocity_7d"] = (df["txn_count_last_24h"] * 3).round().astype(int)
    else:
        df["velocity_1h"], df["velocity_24h"], df["velocity_7d"] = 0, 0, 0

    if "customer_tenure_days" not in df.columns: df["customer_tenure_days"] = 0
    if "location_risk_score" not in df.columns: df["location_risk_score"] = df.get("location_change_flag", 0)
    if "transaction_success_rate_customer" not in df.columns: df["transaction_success_rate_customer"] = df.get("otp_success_rate_customer", 1.0)
    if "ip_address_country_match" not in df.columns: df["ip_address_country_match"] = 1
    if "previous_chargeback_count" not in df.columns: df["previous_chargeback_count"] = df.get("past_disputes_customer", 0)
    if "time_of_day" not in df.columns and "hour_of_day" in df.columns: df["time_of_day"] = df["hour_of_day"]
    
    # 2. Historical Customer Average Amount
    df["historical_avg_amount"] = (
        df.groupby("customer_id")["amount"]
        .transform(lambda x: x.shift().expanding().mean())
    )
    # For a customer's first transaction, there is no history. Fill with current amount (deviation = 0)
    df["historical_avg_amount"] = df["historical_avg_amount"].fillna(df["amount"])
    df["amount_deviation"] = df["amount"] - df["historical_avg_amount"]

    # 3. Cumulative Device Sharing 
    # How many unique customers have used this device SO FAR?
    # Mark True only the first time a customer_id uses a device_id
    df['is_first_device_use'] = ~df.duplicated(subset=['device_id', 'customer_id'])
    # Cumulative sum of new customers per device over time
    df['device_customer_sharing'] = df.groupby('device_id')['is_first_device_use'].cumsum()

    # --- Ratios & Composite Scores ---
    df["velocity_ratio_1h_24h"] = df["velocity_1h"] / (df["velocity_24h"] + 1e-3)
    df["velocity_ratio_24h_7d"] = df["velocity_24h"] / (df["velocity_7d"] + 1e-3)

    df["combined_risk_index"] = (
        0.3 * df["ip_address_risk_score"]
        + 0.2 * (1 - df["device_trust_score"])
        + 0.2 * df["historical_fraud_rate"]
        + 0.2 * df["merchant_historical_fraud_rate"]
        + 0.1 * df.get("location_risk_score", 0.0)
    )

    # Cleanup temporary columns
    df = df.drop(columns=['historical_avg_amount', 'is_first_device_use'], errors='ignore')

    # Restore original index order so we don't scramble downstream arrays
    df = df.sort_index()

    return df


def build_preprocessing_pipeline() -> ImbPipeline:
    """
    Builds an imbalanced-learn pipeline containing the preprocessor and SMOTE.
    """
    logger.info("Constructing ColumnTransformer and SMOTE pipeline...")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ALL_NUMERIC_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
        ]
    )

    # imblearn pipeline automatically skips SMOTE during inference (transform/predict)
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42))
    ])
    
    return pipeline