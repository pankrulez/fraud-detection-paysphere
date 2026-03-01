import numpy as np
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px

from ..core.data_loader import load_sample_data
from ..core.model_loader import load_default_model
from ..core.risk_engine import compute_metrics
from ..core.cache_engine import get_cached_probabilities


def layout(global_state):

    df, probs = get_cached_probabilities()

    if probs.ndim == 2:
        probs = probs[:, 1]

    metrics = compute_metrics(
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

    kpis = dbc.Row([
        dbc.Col(kpi_card("Precision", f"{metrics['precision']:.2%}"), width=3),
        dbc.Col(kpi_card("Recall", f"{metrics['recall']:.2%}"), width=3),
        dbc.Col(kpi_card("Total Cost", f"₹{metrics['total_cost']:,.0f}"), width=3),
        dbc.Col(kpi_card("Cost / 1K Txn", f"₹{metrics['cost_per_1000']:,.0f}"), width=3),
    ], className="g-4")

    decision_counts = {
        "Allow": metrics["tn"] + metrics["fn"],
        "Review": metrics["review"],
        "Block": metrics["tp"] + metrics["fp"],
    }

    fig_decision = px.pie(
        names=list(decision_counts.keys()),
        values=list(decision_counts.values()),
        hole=0.65,
    )

    fig_decision.update_traces(
        textinfo="percent",
        textfont_size=14,
        marker=dict(
            line=dict(width=0)
        )
    )

    fig_decision.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
        height=450
    )

    return html.Div([

        html.H2(
            "Executive Risk Intelligence",
            style={
                "fontSize": "28px",
                "fontWeight": "600",
                "marginBottom": "30px"
            }
        ),

        kpis,

        html.Div(style={"height": "40px"}),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_decision), width=6),
            dbc.Col(html.Div(), width=6),  # future chart placeholder
        ]),

    ])


def kpi_card(title, value):
    return dbc.Card(
        dbc.CardBody([
            html.Div(
                title,
                style={
                    "fontSize": "11px",
                    "opacity": 0.6,
                    "textTransform": "uppercase",
                    "letterSpacing": "1px",
                },
            ),
            html.Div(
                value,
                style={
                    "fontSize": "30px",
                    "fontWeight": "600",
                    "marginTop": "6px",
                },
            ),
        ]),
        style={
            "height": "110px",
            "display": "flex",
            "justifyContent": "center",
        },
    )
