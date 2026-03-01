from dash import html, dcc
import dash_bootstrap_components as dbc

def base_layout(sidebar):

    return html.Div([
        dcc.Location(id="url"),
        dcc.Store(id="global-state"),

        dbc.Row([
            dbc.Col(
                sidebar,
                width=2,
                style={
                    "backgroundColor": "#0f172a",
                    "minHeight": "100vh",
                    "padding": "24px",
                    "borderRight": "1px solid rgba(255,255,255,0.05)"
                }
            ),
            dbc.Col(
                html.Div(
                    id="page-content",
                    style={
                        "padding": "40px",
                        "maxWidth": "1400px",
                        "margin": "0 auto"
                    }
                ),
                width=10
            ),
        ], className="g-0")
    ])