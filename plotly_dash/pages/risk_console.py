from dash import html, dcc
import dash_bootstrap_components as dbc
import numpy as np

from ..core.model_loader import load_default_model


def layout(state):

    return dbc.Container([

        html.H2("Real-Time Risk Console"),

        dbc.Row([
            dbc.Col([
                html.Label("Transaction Amount"),
                dbc.Input(id="amount-input", type="number", value=500),

                html.Br(),
                dbc.Button("Score Transaction", id="score-btn"),

            ], width=4),

            dbc.Col([
                html.H4("Fraud Probability"),
                html.Div(id="prob-output", style={"fontSize": "30px"}),

                html.Br(),
                html.H4("Decision"),
                html.Div(id="decision-output", style={"fontSize": "26px"}),
            ], width=6)
        ])

    ], fluid=True)