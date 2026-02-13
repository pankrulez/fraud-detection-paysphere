import pandas as pd
from typing import Tuple
from src.logger import get_logger
from src.utils.io_utils import load_model
from src.features.feature_engineering import prepare_features

logger = get_logger(__name__)


class FraudScorer:
    """
    Wraps model + preprocessing for real-time scoring.
    """

    def __init__(
        self,
        model_path: str = "models/artifacts/fraud_model.joblib",
        encoders_path: str = "models/encoders/preprocessing.joblib",
        threshold: float = 0.5,
    ):
        self.model = load_model(model_path)
        self.encoders = load_model(encoders_path)
        self.threshold = threshold

    def predict_proba(self, df_txn: pd.DataFrame) -> float:
        """
        Given a single-transaction DataFrame, return fraud probability.
        """
        X, _, _ = prepare_features(
            df_txn.copy(),
            target_col="is_fraud",  # will be dropped if absent
            fit=False,
            encoders=self.encoders,
        )
        prob = self.model.predict_proba(X)[:, 1][0]
        logger.info(f"Fraud probability: {prob:.4f}")
        return prob

    def predict_label_and_action(self, df_txn: pd.DataFrame) -> Tuple[int, str]:
        """
        Returns:
            label: 1 = fraud, 0 = genuine
            action: recommended operational action
        """
        p = self.predict_proba(df_txn)
        label = int(p >= self.threshold)

        # Simple decision policy:
        # - p >= 0.9: hard block
        # - 0.7 <= p < 0.9: OTP challenge or manual review
        # - 0.5 <= p < 0.7: soft review or step-up authentication
        # - else: allow
        if p >= 0.9:
            action = "HARD_BLOCK"
        elif p >= 0.7:
            action = "OTP_CHALLENGE"
        elif p >= 0.5:
            action = "SOFT_REVIEW"
        else:
            action = "ALLOW"

        logger.info(f"Predicted label={label}, action={action}")
        return label, action