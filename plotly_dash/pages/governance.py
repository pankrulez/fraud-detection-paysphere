from dash import html
import dash_bootstrap_components as dbc


def layout(state):

    return dbc.Container([
        html.H2("Model Governance"),

        dbc.Card(
            dbc.CardBody([
                html.P("Model: fraud_model.joblib"),
                html.P("Version: 1.0"),
                html.P("Threshold Allow: " + str(state["threshold_allow"])),
                html.P("Threshold Block: " + str(state["threshold_block"])),
            ])
        )
    ], fluid=True)