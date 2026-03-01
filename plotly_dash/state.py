from .config import *

def initial_state():
    return {
        "threshold_allow": DEFAULT_THRESHOLD_ALLOW,
        "threshold_block": DEFAULT_THRESHOLD_BLOCK,
        "cost_fp": DEFAULT_COST_FP,
        "cost_fn": DEFAULT_COST_FN,
        "review_cost": DEFAULT_REVIEW_COST,
        "recovery_rate": DEFAULT_RECOVERY_RATE,
        "revenue_per_txn": DEFAULT_REVENUE_PER_TXN,
        "model_version": "default",
    }