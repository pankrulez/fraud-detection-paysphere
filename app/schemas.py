# app/schemas.py
from dataclasses import dataclass
from typing import Literal, Optional
from datetime import datetime


PaymentMethod = Literal["UPI", "CARD", "NETBANKING", "WALLET"]
MerchantCategory = Literal["Electronics", "Travel", "Fashion", "Gaming", "Grocery", "Utilities"]


@dataclass
class TransactionSchema:
    """
    Business-level schema for a single transaction.

    Mirrors the columns from the data definition, with business meaning:
    - transaction_id: unique transaction key, used for traceability.
    - customer_id: customer identifier, key for behavioural features.
    - device_id: device fingerprint, helps detect risky device reuse.
    - merchant_id: merchant identifier, used for merchant risk features.
    - timestamp: transaction time for temporal and velocity analysis.
    - amount: monetary value, different bands show distinct fraud patterns.
    - payment_method: UPI / CARD / NETBANKING / WALLET.
    - is_international: 1 if cross-border payment, higher inherent risk.
    - merchant_category: domain (e.g., Travel, Gaming) with different risk.
    - ip_address_risk_score: 0–1 proxy for IP reputation / proxy usage.
    - device_trust_score: 0–1 trust metric; low values suggest emulators.
    - velocity_1h / 24h / 7d: behavioural transaction velocity windows.
    - customer_tenure_days: account age; very new accounts are riskier.
    - historical_fraud_rate: customer-level past fraud ratio.
    - merchant_historical_fraud_rate: merchant’s past fraud ratio.
    - ip_address_country_match: 1 if IP country matches customer country.
    - previous_chargeback_count: prior chargebacks for this customer.
    - time_of_day / day_of_week / is_weekend: temporal risk indicators.
    - location_risk_score: 0–1 risk of geo / region.
    - transaction_success_rate_customer: past success ratio, low suggests trial-and-error.
    - is_fraud: target label, 1 = fraud, 0 = genuine.
    """

    transaction_id: int
    customer_id: int
    device_id: int
    merchant_id: int
    timestamp: datetime

    amount: float
    payment_method: PaymentMethod
    is_international: int
    merchant_category: MerchantCategory

    ip_address_risk_score: float
    device_trust_score: float

    velocity_1h: int
    velocity_24h: int
    velocity_7d: int

    customer_tenure_days: int
    historical_fraud_rate: float
    merchant_historical_fraud_rate: float

    ip_address_country_match: int
    previous_chargeback_count: int

    time_of_day: int
    day_of_week: int
    is_weekend: int

    location_risk_score: float
    transaction_success_rate_customer: float

    is_fraud: Optional[int] = 0  # default 0 for inference


def to_dataframe_dict(txn: "TransactionSchema") -> dict:
    """
    Convert TransactionSchema into a dict aligned with training columns.
    """
    return {
        "transaction_id": txn.transaction_id,
        "customer_id": txn.customer_id,
        "device_id": txn.device_id,
        "merchant_id": txn.merchant_id,
        "timestamp": txn.timestamp.isoformat(),
        "amount": txn.amount,
        "payment_method": txn.payment_method,
        "is_international": txn.is_international,
        "merchant_category": txn.merchant_category,
        "ip_address_risk_score": txn.ip_address_risk_score,
        "device_trust_score": txn.device_trust_score,
        "velocity_1h": txn.velocity_1h,
        "velocity_24h": txn.velocity_24h,
        "velocity_7d": txn.velocity_7d,
        "customer_tenure_days": txn.customer_tenure_days,
        "historical_fraud_rate": txn.historical_fraud_rate,
        "merchant_historical_fraud_rate": txn.merchant_historical_fraud_rate,
        "ip_address_country_match": txn.ip_address_country_match,
        "previous_chargeback_count": txn.previous_chargeback_count,
        "time_of_day": txn.time_of_day,
        "day_of_week": txn.day_of_week,
        "is_weekend": txn.is_weekend,
        "location_risk_score": txn.location_risk_score,
        "transaction_success_rate_customer": txn.transaction_success_rate_customer,
        "is_fraud": txn.is_fraud,
    }