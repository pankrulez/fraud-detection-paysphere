from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px

from ..core.cache_engine import get_cached_probabilities


def layout(state):

    df, probs = get_cached_probabilities()

    df["prob"] = probs

    fig = px.histogram(
        df,
        x="prob",
        nbins=50,
        title="Fraud Probability Distribution"
    )

    return dbc.Container([
        html.H2("Risk Segment Explorer"),
        dcc.Graph(figure=fig),
    ], fluid=True)