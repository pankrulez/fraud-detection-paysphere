from dash import html
import dash_bootstrap_components as dbc

from ..core.cache_engine import get_cached_probabilities
from ..core.risk_engine import compute_metrics


def layout(state):

    df, probs = get_cached_probabilities()

    metrics = compute_metrics(
        df,
        probs,
        state["threshold_allow"],
        state["threshold_block"],
        state["cost_fp"],
        state["cost_fn"],
        state["review_cost"],
        state["recovery_rate"],
        state["revenue_per_txn"],
    )

    return dbc.Container([
        html.H2("Executive Risk Intelligence"),

        dbc.Row([
            dbc.Col(kpi("Precision", f"{metrics['precision']:.2%}")),
            dbc.Col(kpi("Recall", f"{metrics['recall']:.2%}")),
            dbc.Col(kpi("Total Cost", f"₹{metrics['total_cost']:,.0f}")),
            dbc.Col(kpi("Cost / 1K", f"₹{metrics['cost_per_1000']:,.0f}")),
        ])
    ], fluid=True)


def kpi(title, value):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, style={"opacity": 0.6}),
            html.H3(value)
        ])
    )