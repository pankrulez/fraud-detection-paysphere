from functools import lru_cache
import joblib
import numpy as np
from ..config import DEFAULT_MODEL_PATH, DEFAULT_ENCODER_PATH
from src.modeling.inference import FraudScorer

@lru_cache()
def load_default_model():
    return FraudScorer(
        model_path=DEFAULT_MODEL_PATH,
        encoders_path=DEFAULT_ENCODER_PATH,
        threshold=0.5
    )

def load_uploaded_model(model_path, encoder_path):
    return FraudScorer(
        model_path=model_path,
        encoders_path=encoder_path,
        threshold=0.5
    )
    
def extract_positive_class(probs):
        probs = np.asarray(probs)

        if probs.ndim == 2:
            return probs[:, 1]
        return probs