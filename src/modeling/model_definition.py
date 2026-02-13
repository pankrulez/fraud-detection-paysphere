from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_model(algorithm: str = "xgboost"):
    """
    Factory method to create a fraud classification model.

    Defaults to gradient-boosted trees which often work well on tabular fraud data.
    """
    if algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=42,
        )
    elif algorithm == "xgboost":
        return XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
            scale_pos_weight=1.0,  # can be tuned instead of SMOTE
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")