from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from datetime import datetime
import uuid
from typing import List

from src.modeling.inference import FraudScorer
from src.logger import get_logger, setup_logging

setup_logging(default_path="config/logging.yaml")
# logger = get_logger(__name__)
logger = get_logger("fraud")

app = FastAPI(
    title="PaySphere Risk API", 
    version="1.0.0",
    description="Real-time fraud scoring API for digital payments."
)
@app.on_event("startup")
async def startup_event():
    logger.info("Risk Engine API is starting up...")

# Initialize the scorer once on startup
try:
    logger.info("Initializing FraudScorer...")
    scorer = FraudScorer()
    logger.info("FraudScorer initialized successfully.")
except Exception as e:
    logger.error(f"Failed to load model on startup: {e}")
    raise RuntimeError("Model initialization failed. Check artifacts.")


class TransactionRequest(BaseModel):
    """
    The data contract for incoming transactions. 
    These are the RAW features before our pipeline engineers them.
    """
    customer_id: str = Field(..., description="Unique customer identifier")
    device_id: str = Field(..., description="Unique device identifier")
    merchant_id: str = Field(..., description="Unique merchant identifier")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    amount: float = Field(..., gt=0, description="Transaction amount")
    payment_method: str = Field(..., description="e.g., credit_card, debit_card, upi")
    merchant_category: str = Field(..., description="e.g., electronics, travel, food")
    
    ip_address_risk_score: float = Field(0.0, ge=0.0, le=1.0)
    device_trust_score: float = Field(1.0, ge=0.0, le=1.0)
    
    is_international: int = Field(0, description="1 if cross-border, 0 otherwise")
    is_weekend: int = Field(0, description="1 if weekend, 0 otherwise")
    
    # Contextual fields
    past_fraud_count_customer: int = Field(0)
    past_disputes_customer: int = Field(0)
    txn_count_last_24h: int = Field(0)
    customer_tenure_days: int = Field(0)
    location_change_flag: int = Field(0)
    otp_success_rate_customer: float = Field(1.0)
    ip_address_country_match: int = Field(1)
    hour_of_day: int = Field(12, ge=0, le=23)
    
    # --- THE TWO MISSING FIELDS ---
    day_of_week: int = Field(0, ge=0, le=6)
    merchant_historical_fraud_rate: float = Field(0.0, ge=0.0, le=1.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "CUST_9876",
                "device_id": "DEV_1234",
                "merchant_id": "MERCH_555",
                "amount": 1250.50,
                "payment_method": "credit_card",
                "merchant_category": "electronics",
                "ip_address_risk_score": 0.85,
                "device_trust_score": 0.2,
                "is_international": 1,
                "is_weekend": 0,
                "txn_count_last_24h": 5,
                "hour_of_day": 3,
                "day_of_week": 2,
                "merchant_historical_fraud_rate": 0.15
            }
        }
    }


@app.get("/health")
async def health_check():
    """Simple health check endpoint for load balancers."""
    return {"status": "healthy", "model_loaded": scorer is not None}


@app.post("/v1/score")
async def get_fraud_score(txn: TransactionRequest):
    try:
        # 1. Convert payload to DataFrame using modern Pydantic method
        df_txn = pd.DataFrame([txn.model_dump()])
        
        # 2. Get label, action, and probability in a SINGLE inference call
        label, action, prob = scorer.predict_label_and_action(df_txn)
        
        # 3. Generate a unique trace ID for logging/auditing
        trace_id = f"TXN_{uuid.uuid4().hex[:8].upper()}"
        
        logger.info(f"[{trace_id}] Scored transaction. Action: {action}, Prob: {prob:.4f}")
        
        return {
            "transaction_id": trace_id,
            "fraud_probability": round(prob, 4),
            "risk_label": label,
            "recommended_action": action,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error processing transaction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal ML processing error.")
    
@app.post("/v1/batch-score")
async def get_batch_score(txns: List[TransactionRequest]):
    """Receives a batch of transactions and scores them in one go."""
    try:
        # Convert list of Pydantic models to a single DataFrame
        df_txns = pd.DataFrame([t.model_dump() for t in txns])
        
        # Score the entire DataFrame at once
        probs = scorer.predict_proba_batch(df_txns)
        
        return {"status": "success", "probabilities": probs}
    except Exception as e:
        logger.error(f"Batch error: {str(e)}")
        raise HTTPException(status_code=500, detail="Batch processing error.")