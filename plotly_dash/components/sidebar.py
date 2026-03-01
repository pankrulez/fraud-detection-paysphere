from dash import html, dcc
import dash_bootstrap_components as dbc

def build_sidebar(initial):

    return html.Div([
        html.H4(
            "Fraud Decision Intelligence",
            style={
                "fontSize": "16px",
                "fontWeight": "600",
                "marginBottom": "30px"
            }
        ),

        dbc.Nav(
            [
                dbc.NavLink("Executive", href="/"),
                dbc.NavLink("Model Lab", href="/model"),
                dbc.NavLink("Risk Console", href="/console"),
                dbc.NavLink("Segments", href="/segments"),
                dbc.NavLink("Governance", href="/governance"),
            ],
            vertical=True,
            pills=True,
            style={"gap": "6px"}
        ),

        html.Hr(),

        html.Label("Allow Threshold"),
        dcc.Slider(0.0, 0.5, 0.01,
                   value=initial["threshold_allow"],
                   id="threshold-allow"),

        html.Label("Block Threshold"),
        dcc.Slider(0.05, 0.9, 0.01,
                   value=initial["threshold_block"],
                   id="threshold-block"),

        html.Hr(),

        html.Label("Cost per False Positive"),
        dcc.Input(id="cost-fp", type="number",
                  value=initial["cost_fp"]),

        html.Label("Cost per False Negative"),
        dcc.Input(id="cost-fn", type="number",
                  value=initial["cost_fn"]),

        html.Label("Review Cost"),
        dcc.Input(id="review-cost", type="number",
                  value=initial["review_cost"]),

        html.Label("Recovery Rate"),
        dcc.Slider(0, 1, 0.01,
                   value=initial["recovery_rate"],
                   id="recovery-rate"),

        html.Label("Revenue per Txn"),
        dcc.Input(id="revenue-per-txn", type="number",
                  value=initial["revenue_per_txn"]),
    ],
    style={
        "padding": "20px",
        "backgroundColor": "#0f172a",
        "height": "100vh"
    }
    )