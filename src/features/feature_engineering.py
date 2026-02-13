from typing import Tuple, List, Dict, Any
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

from src.logger import get_logger

logger = get_logger(__name__)


CATEGORICAL_COLS = ["payment_method", "merchant_category"]
BINARY_COLS = ["is_international", "ip_address_country_match", "is_weekend"]
NUMERIC_COLS = [
    "amount",
    "ip_address_risk_score",
    "device_trust_score",
    "velocity_1h",
    "velocity_24h",
    "velocity_7d",
    "customer_tenure_days",
    "historical_fraud_rate",
    "merchant_historical_fraud_rate",
    "previous_chargeback_count",
    "time_of_day",
    "day_of_week",
    "location_risk_score",
    "transaction_success_rate_customer",
]


def engineer_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional behavioural and risk-based features from the base fields.

    Examples:
    - amount_deviation: deviation from customer's historical average.
    - velocity_ratio_1h_24h: short-term vs daily velocity.
    """
    
    # --- Derive historical_fraud_rate if not present ---
    if "historical_fraud_rate" not in df.columns:
        # Approximate: fraud or disputes over total interactions (fraud + disputes + 1 to avoid zero)
        fraud = df.get("past_fraud_count_customer", 0)
        disputes = df.get("past_disputes_customer", 0)
        df["historical_fraud_rate"] = (fraud + disputes) / (fraud + disputes + 1.0)
        
        # ----- derive “velocity_*” from txn_count_last_24h as a proxy -----
    if "txn_count_last_24h" in df.columns:
        df["velocity_24h"] = df["txn_count_last_24h"]
        df["velocity_1h"] = (df["txn_count_last_24h"] / 6).round().astype(int)
        df["velocity_7d"] = (df["txn_count_last_24h"] * 3).round().astype(int)
    else:
        df["velocity_1h"] = 0
        df["velocity_24h"] = 0
        df["velocity_7d"] = 0

    # customer_tenure_days – placeholder if you do not have account‑creation date
    if "customer_tenure_days" not in df.columns:
        df["customer_tenure_days"] = 0

    # location_risk_score – derive simple proxy from location_change_flag if needed
    if "location_risk_score" not in df.columns:
        df["location_risk_score"] = df.get("location_change_flag", 0)

    # transaction_success_rate_customer – proxy from otp_success_rate_customer
    if "transaction_success_rate_customer" not in df.columns:
        df["transaction_success_rate_customer"] = df.get("otp_success_rate_customer", 1.0)

    # ip_address_country_match / previous_chargeback_count – simple defaults
    if "ip_address_country_match" not in df.columns:
        df["ip_address_country_match"] = 1
    if "previous_chargeback_count" not in df.columns:
        df["previous_chargeback_count"] = df.get("past_disputes_customer", 0)

    # time_of_day – map from hour_of_day if needed
    if "time_of_day" not in df.columns and "hour_of_day" in df.columns:
        df["time_of_day"] = df["hour_of_day"]
    
    
    # Customer-level average amount
    customer_avg = df.groupby("customer_id")["amount"].transform("mean")
    df["amount_deviation"] = df["amount"] - customer_avg

    # Velocity ratios
    df["velocity_ratio_1h_24h"] = df["velocity_1h"] / (df["velocity_24h"] + 1e-3)
    df["velocity_ratio_24h_7d"] = df["velocity_24h"] / (df["velocity_7d"] + 1e-3)

    # Device behavior: how many unique customers per device (proxy for shared devices)
    device_counts = df.groupby("device_id")["customer_id"].transform("nunique")
    df["device_customer_sharing"] = device_counts

    # Composite risk index: simple weighted sum as an example
    df["combined_risk_index"] = (
        0.3 * df["ip_address_risk_score"]
        + 0.2 * (1 - df["device_trust_score"])
        + 0.2 * df["historical_fraud_rate"]
        + 0.2 * df["merchant_historical_fraud_rate"]
        + 0.1 * df.get("location_risk_score", 0.0)
    )

    return df


def prepare_features(
    df: pd.DataFrame, target_col: str, fit: bool = True,
    encoders: Dict[str, Any] | None = None
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    """
    Prepare the feature matrix X and target y.

    - Applies feature engineering.
    - One-hot encodes categorical variables.
    - Standard scales numeric variables.
    - Optionally fits and returns encoders/scalers, or reuses provided ones.

    Returns:
        X: feature DataFrame
        y: target Series
        encoders: dict with "ohe" and "scaler"
    """
    df = engineer_behavioral_features(df)

    # Split target
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Keep ID columns for potential logging, but not for modeling
    id_cols = ["transaction_id", "customer_id", "device_id", "merchant_id", "timestamp"]
    X_model = X.drop(columns=id_cols, errors="ignore")

    # Handle categorical features
    if encoders is None:
        encoders = {}

    if fit:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_encoded = ohe.fit_transform(X_model[CATEGORICAL_COLS])
        encoders["ohe"] = ohe
    else:
        ohe = encoders["ohe"]  # type: ignore
        cat_encoded = ohe.transform(X_model[CATEGORICAL_COLS])

    cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_COLS)
    X_cat = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=X_model.index)

    # Numeric + binary
    X_num = X_model[NUMERIC_COLS + BINARY_COLS + [
        "amount_deviation",
        "velocity_ratio_1h_24h",
        "velocity_ratio_24h_7d",
        "device_customer_sharing",
        "combined_risk_index",
    ]]

    if fit:
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)
        encoders["scaler"] = scaler
    else:
        scaler = encoders["scaler"]  # type: ignore
        X_num_scaled = scaler.transform(X_num)

    X_num_df = pd.DataFrame(X_num_scaled, columns=X_num.columns, index=X_num.index)

    X_final = pd.concat([X_num_df, X_cat], axis=1)
    logger.info(f"Final feature matrix shape: {X_final.shape}")

    return X_final, y, encoders


def handle_imbalance(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handles class imbalance using SMOTE on the minority class (fraud).
    """
    logger.info("Applying SMOTE to handle class imbalance.")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)  # type: ignore
    X_res = pd.DataFrame(X_res, columns=X.columns)
    y_res = pd.Series(y_res.to_numpy(), name=y.name)
    logger.info(f"After SMOTE, X shape: {X_res.shape}, fraud rate: {y_res.mean():.4f}")
    return X_res, y_res