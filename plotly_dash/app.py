from dash import Dash, Input, Output, State, dcc
import dash_bootstrap_components as dbc
import plotly.io as pio
import plotly.graph_objects as go

from .state import initial_state
from .components.layout import base_layout
from .components.sidebar import build_sidebar
from .pages.executive import layout as executive_layout
from .pages.model_lab import layout as model_lab_layout

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)

pio.templates["fintech_dark"] = go.layout.Template(
    layout=dict(
        paper_bgcolor="#0b0f17",
        plot_bgcolor="#0b0f17",
        font=dict(color="#e5e7eb"),
        margin=dict(l=20, r=20, t=40, b=20)
    )
)

pio.templates.default = "fintech_dark"

server = app.server

init = initial_state()

app.layout = base_layout(build_sidebar(init))


# ---------------------------------
# Update Global State From Sidebar
# ---------------------------------
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
def update_global_state(
    t_allow,
    t_block,
    cost_fp,
    cost_fn,
    review_cost,
    recovery_rate,
    revenue,
):
    return {
        "threshold_allow": t_allow,
        "threshold_block": t_block,
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
        "review_cost": review_cost,
        "recovery_rate": recovery_rate,
        "revenue_per_txn": revenue,
    }


# ---------------------------------
# Routing
# ---------------------------------
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

    return executive_layout(state)


if __name__ == "__main__":
    app.run(debug=True)