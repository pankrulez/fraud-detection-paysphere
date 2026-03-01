from src.modeling.inference import FraudScorer


def load_default_model():

    return FraudScorer(
        model_path="models/artifacts/fraud_model.joblib",
        encoders_path="models/encoders/preprocessing.joblib",
        threshold=0.5,
    )