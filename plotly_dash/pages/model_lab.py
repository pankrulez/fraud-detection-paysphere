from dash import html, dcc
import numpy as np
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from ..core.cache_engine import get_cached_probabilities


def layout(state):

    df, probs = get_cached_probabilities()
    y_true = np.asarray(df["is_fraud"])

    # ROC
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig_roc.add_annotation(text=f"AUC = {roc_auc:.3f}", x=0.6, y=0.2)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, probs)

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR"))

    return dbc.Container([
        html.H2("Model Intelligence Lab"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_roc), width=6),
            dbc.Col(dcc.Graph(figure=fig_pr), width=6),
        ])
    ], fluid=True)