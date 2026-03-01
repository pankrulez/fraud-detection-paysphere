import numpy as np
import pandas as pd

def compute_metrics(df, probs,
                    threshold_allow,
                    threshold_block,
                    cost_fp,
                    cost_fn,
                    review_cost,
                    recovery_rate,
                    revenue_per_txn):

    df = df.copy()
    df["prob"] = probs

    conditions = [
        df["prob"] < threshold_allow,
        (df["prob"] >= threshold_allow) & (df["prob"] < threshold_block),
        df["prob"] >= threshold_block,
    ]

    decisions = np.select(
        conditions,
        ["allow", "review", "block"],
        default="allow"
    ).astype(object)
    df["decision"] = decisions

    tp = ((df["decision"] == "block") & (df["is_fraud"] == 1)).sum()
    fp = ((df["decision"] == "block") & (df["is_fraud"] == 0)).sum()
    fn = ((df["decision"] == "allow") & (df["is_fraud"] == 1)).sum()
    tn = ((df["decision"] == "allow") & (df["is_fraud"] == 0)).sum()
    review_count = (df["decision"] == "review").sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    fp_cost = fp * cost_fp
    fn_cost = fn * cost_fn * (1 - recovery_rate)
    review_total_cost = review_count * review_cost
    revenue_loss = fp * revenue_per_txn

    total_cost = fp_cost + fn_cost + review_total_cost + revenue_loss

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "review": review_count,
        "precision": precision,
        "recall": recall,
        "total_cost": total_cost,
        "cost_per_1000": total_cost / len(df) * 1000,
    }
    

def threshold_cost_curve(
    df,
    probs,
    cost_fp,
    cost_fn,
    review_cost,
    recovery_rate,
    revenue_per_txn,
    n_points=100,
):

    thresholds = np.linspace(0.01, 0.9, n_points)
    results = []

    for t in thresholds:

        metrics = compute_metrics(
            df,
            probs,
            threshold_allow=0.0,
            threshold_block=t,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
            review_cost=review_cost,
            recovery_rate=recovery_rate,
            revenue_per_txn=revenue_per_txn,
        )

        results.append(
            {
                "threshold": t,
                "total_cost": metrics["total_cost"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )

    return pd.DataFrame(results)