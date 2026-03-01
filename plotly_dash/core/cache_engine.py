import numpy as np
from functools import lru_cache
from .data_loader import load_sample_data
from .model_loader import load_default_model


@lru_cache(maxsize=1)
def get_cached_probabilities():

    df = load_sample_data()
    scorer = load_default_model()

    raw_probs = scorer.predict_proba(df)
    probs = np.asarray(raw_probs)

    # Guarantee vector shape
    if probs.ndim == 0:
        probs = np.full(len(df), float(probs))
    elif probs.ndim == 2:
        probs = probs[:, 1]

    return df, probs