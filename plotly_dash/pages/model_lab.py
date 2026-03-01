import numpy as np
import pandas as pd

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    auc,
)

from ..core.cache_engine import get_cached_probabilities
from ..core.risk_engine import compute_metrics, threshold_cost_curve


def layout(global_state):

    # --------------------------------------------------
    # Load Cached Data + Probabilities (Heavy compute once)
    # --------------------------------------------------
    df, probs = get_cached_probabilities()
    y_true = np.asarray(df["is_fraud"])

    # --------------------------------------------------
    # Current Metrics (based on operating thresholds)
    # --------------------------------------------------
    metrics_current = compute_metrics(
        df,
        probs,
        global_state["threshold_allow"],
        global_state["threshold_block"],
        global_state["cost_fp"],
        global_state["cost_fn"],
        global_state["review_cost"],
        global_state["recovery_rate"],
        global_state["revenue_per_txn"],
    )

    # --------------------------------------------------
    # ROC Curve
    # --------------------------------------------------
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash"),
            name="Random",
        )
    )
    fig_roc.update_layout(
        title=f"ROC Curve (AUC = {roc_auc:.4f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )

    # --------------------------------------------------
    # Precision–Recall Curve
    # --------------------------------------------------
    precision, recall, _ = precision_recall_curve(y_true, probs)

    fig_pr = go.Figure()
    fig_pr.add_trace(
        go.Scatter(x=recall, y=precision, mode="lines", name="PR Curve")
    )
    fig_pr.update_layout(
        title="Precision–Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
    )

    # --------------------------------------------------
    # Threshold vs Cost Curve
    # --------------------------------------------------
    cost_df = threshold_cost_curve(
        df,
        probs,
        global_state["cost_fp"],
        global_state["cost_fn"],
        global_state["review_cost"],
        global_state["recovery_rate"],
        global_state["revenue_per_txn"],
    )

    optimal_row = cost_df.loc[cost_df["total_cost"].idxmin()]
    optimal_threshold = optimal_row["threshold"]
    optimal_cost = optimal_row["total_cost"]

    fig_cost = go.Figure()
    fig_cost.add_trace(
        go.Scatter(
            x=cost_df["threshold"],
            y=cost_df["total_cost"],
            mode="lines",
            name="Total Cost",
        )
    )

    # Optimal threshold marker
    fig_cost.add_trace(
        go.Scatter(
            x=[optimal_threshold],
            y=[optimal_cost],
            mode="markers",
            marker=dict(size=10),
            name="Optimal Threshold",
        )
    )

    # Current threshold marker
    current_cost = cost_df.loc[
        (cost_df["threshold"] - global_state["threshold_block"]).abs().idxmin(),
        "total_cost",
    ]

    fig_cost.add_trace(
        go.Scatter(
            x=[global_state["threshold_block"]],
            y=[current_cost],
            mode="markers",
            marker=dict(size=10),
            name="Current Threshold",
        )
    )

    fig_cost.update_layout(
        title="Threshold vs Business Cost",
        xaxis_title="Block Threshold",
        yaxis_title="Total Cost",
    )

    # --------------------------------------------------
    # Threshold vs Precision & Recall
    # --------------------------------------------------
    thresholds = np.linspace(0.01, 0.9, 100)
    precision_list = []
    recall_list = []

    for t in thresholds:
        metrics = compute_metrics(
            df,
            probs,
            threshold_allow=0.0,
            threshold_block=t,
            cost_fp=global_state["cost_fp"],
            cost_fn=global_state["cost_fn"],
            review_cost=global_state["review_cost"],
            recovery_rate=global_state["recovery_rate"],
            revenue_per_txn=global_state["revenue_per_txn"],
        )

        precision_list.append(metrics["precision"])
        recall_list.append(metrics["recall"])

    fig_tradeoff = go.Figure()
    fig_tradeoff.add_trace(
        go.Scatter(
            x=thresholds,
            y=precision_list,
            mode="lines",
            name="Precision",
        )
    )
    fig_tradeoff.add_trace(
        go.Scatter(
            x=thresholds,
            y=recall_list,
            mode="lines",
            name="Recall",
        )
    )

    fig_tradeoff.update_layout(
        title="Threshold vs Precision & Recall",
        xaxis_title="Block Threshold",
        yaxis_title="Metric Value",
    )

    # --------------------------------------------------
    # Confusion Matrix (Binary at Block Threshold)
    # --------------------------------------------------
    preds = (probs >= global_state["threshold_block"]).astype(int)
    cm = confusion_matrix(y_true, preds)

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=["Genuine", "Fraud"],
        y=["Genuine", "Fraud"],
        title="Confusion Matrix",
    )

    # --------------------------------------------------
    # KPI Strip
    # --------------------------------------------------
    kpi_row = dbc.Row([
        dbc.Col(kpi_card("Current Threshold", f"{global_state['threshold_block']:.3f}")),
        dbc.Col(kpi_card("Optimal Threshold", f"{optimal_threshold:.3f}")),
        dbc.Col(kpi_card("Current Cost", f"₹{metrics_current['total_cost']:,.0f}")),
        dbc.Col(kpi_card("Optimal Cost", f"₹{optimal_cost:,.0f}")),
    ])

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    return dbc.Container([

        html.H2("Model Intelligence Lab"),
        html.Hr(),

        kpi_row,
        html.Hr(),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_roc), width=6),
            dbc.Col(dcc.Graph(figure=fig_pr), width=6),
        ]),

        html.Hr(),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_tradeoff), width=6),
            dbc.Col(dcc.Graph(figure=fig_cost), width=6),
        ]),

        html.Hr(),

        dcc.Graph(figure=fig_cm),

    ], fluid=True)


# --------------------------------------------------
# KPI Card Component
# --------------------------------------------------

def kpi_card(title, value):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, style={"fontSize": "13px", "opacity": 0.6}),
            html.H4(value)
        ])
    )