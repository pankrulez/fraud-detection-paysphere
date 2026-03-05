import pandas as pd
from typing import Tuple
from src.logger import get_logger
from src.utils.io_utils import load_model
from src.features.feature_engineering import engineer_behavioral_features

logger = get_logger(__name__)

class FraudScorer:
    def __init__(self, pipeline_path="models/artifacts/fraud_pipeline.joblib", threshold=0.5):
        """
        Loads the unified ImbPipeline (which handles scaling, OHE, and the model).
        """
        self.pipeline = load_model(pipeline_path)
        self.threshold = threshold 

    def predict_proba(self, df_txn: pd.DataFrame) -> float:
        """
        Engineers features and scores the transaction using the unified pipeline.
        """
        try:
            # 1. Engineer behavioral features
            # NOTE: For a single-row live inference without a Feature Store, 
            # historical aggregations will default to the current row's values.
            X_engineered = engineer_behavioral_features(df_txn.copy())
            
            # 2. Drop identifiers (the pipeline expects raw feature columns only)
            id_cols = ["transaction_id", "customer_id", "device_id", "merchant_id", "timestamp"]
            X_model = X_engineered.drop(columns=id_cols, errors="ignore")
            
            # Safety check: ensure no target column slipped through
            if "is_fraud" in X_model.columns:
                X_model = X_model.drop(columns=["is_fraud"])

            # 3. Score the data 
            # The unified pipeline automatically scales and encodes before predicting
            prob = self.pipeline.predict_proba(X_model)[:, 1][0]
            
            logger.info(f"Transaction scored successfully. Probability: {prob:.4f}")
            return prob
            
        except Exception as e:
            logger.error(f"Error during prediction routing: {e}")
            raise
        
    
    def predict_proba_batch(self, df_batch: pd.DataFrame) -> list:
        """Scores a whole batch of transactions instantly."""
        try:
            X_engineered = engineer_behavioral_features(df_batch.copy())
            id_cols = ["transaction_id", "customer_id", "device_id", "merchant_id", "timestamp", "is_fraud"]
            X_model = X_engineered.drop(columns=id_cols, errors="ignore")
            
            # Return all probabilities as a standard Python list
            probs = self.pipeline.predict_proba(X_model)[:, 1]
            return probs.tolist()
        except Exception as e:
            logger.error(f"Batch scoring error: {e}")
            raise
    

    def predict_label_and_action(self, df_txn: pd.DataFrame) -> Tuple[int, str, float]:
        """
        Returns the binary label, the business action, and the raw probability.
        """
        p = self.predict_proba(df_txn)
        label = int(p >= self.threshold)

        # Multi-Tiered Relative Policy based on risk severity
        # Cap the extreme risk multiplier at 0.85 to prevent impossible thresholds
        if p >= min(self.threshold * 5, 0.85):
            action = "HARD_BLOCK"  # Extreme risk
        elif p >= (self.threshold * 3):
            action = "MANUAL_REVIEW" # High risk relative to baseline
        elif p >= self.threshold:
            action = "OTP_VERIFICATION" # Step-up authentication
        else:
            action = "ALLOW" # Safe transaction
            
        txn_id = df_txn['transaction_id'].iloc[0] if 'transaction_id' in df_txn.columns else "N/A"
        logger.info(f"Scoring TXN: {txn_id} | Prob: {p:.4f} | Action: {action}")
            
        return label, action, p