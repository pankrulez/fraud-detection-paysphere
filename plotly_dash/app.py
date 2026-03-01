from dash import Dash, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.io as pio
import plotly.graph_objects as go

from .state import initial_state
from .components.layout import base_layout
from .components.sidebar import build_sidebar

from .pages.executive import layout as executive_layout
from .pages.model_lab import layout as model_lab_layout
from .pages.risk_console import layout as console_layout
from .pages.segment_explorer import layout as segment_layout
from .pages.governance import layout as governance_layout

from .core.model_loader import load_default_model


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)

server = app.server


# -------- Fintech Theme --------
pio.templates["fintech_dark"] = go.layout.Template(
    layout=dict(
        paper_bgcolor="#0b0f17",
        plot_bgcolor="#0b0f17",
        font=dict(color="#e5e7eb"),
    )
)
pio.templates.default = "fintech_dark"


# -------- Layout --------
init = initial_state()
app.layout = base_layout(build_sidebar(init))


# -------- Global State --------
@app.callback(
    Output("global-state", "data"),
    Input("threshold-allow", "value"),
    Input("threshold-block", "value"),
    Input("cost-fp", "value"),
    Input("cost-fn", "value"),
    Input("review-cost", "value"),
    Input("recovery-rate", "value"),
    Input("revenue-per-txn", "value"),
)
def update_state(a, b, cfp, cfn, rc, rr, rev):
    return {
        "threshold_allow": a,
        "threshold_block": b,
        "cost_fp": cfp,
        "cost_fn": cfn,
        "review_cost": rc,
        "recovery_rate": rr,
        "revenue_per_txn": rev,
    }


# -------- Routing --------
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("global-state", "data"),
)
def route(pathname, state):

    if state is None:
        state = initial_state()

    if pathname == "/model":
        return model_lab_layout(state)

    if pathname == "/console":
        return console_layout(state)

    if pathname == "/segments":
        return segment_layout(state)

    if pathname == "/governance":
        return governance_layout(state)

    return executive_layout(state)


if __name__ == "__main__":
    app.run(debug=True)