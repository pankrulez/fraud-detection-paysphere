from functools import lru_cache
import numpy as np
from .data_loader import load_sample_data
from .model_loader import load_default_model

@lru_cache()
def get_cached_probabilities():

    df = load_sample_data()
    scorer = load_default_model()

    raw_probs = scorer.predict_proba(df)
    probs = np.asarray(raw_probs)

    if probs.ndim == 2:
        probs = probs[:, 1]

    return df, probs