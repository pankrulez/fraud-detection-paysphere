import os
import sys
import pandas as pd
import streamlit as st
import plotly.io as pio

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.modeling.inference import FraudScorer
from app.overview_view import render_overview
from app.live_view import render_live_scoring
from app.analytics_view import render_analytics
from app.pipeline_view import render_pipeline

pio.templates.default = "plotly_dark"

st.set_page_config(
    page_title="PaySphere Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- ENTERPRISE THEME ----------
st.markdown("""
<style>
.stApp {
    background: #0f172a;
    color: #e2e8f0;
}

section[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid rgba(255,255,255,0.06);
}

.glass-card {
    background: #111827;
    border-radius: 14px;
    padding: 1.4rem;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
}

[data-testid="stMetric"] {
    background: #111827;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.05);
}

h1, h2, h3 {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_scorer(threshold: float = 0.5):
    return FraudScorer(
        model_path="models/artifacts/fraud_model.joblib",
        encoders_path="models/encoders/preprocessing.joblib",
        threshold=threshold,
    )


@st.cache_data
def load_sample_data(n: int = 50000):
    path = "data/interim/transactions_clean.csv"
    if not os.path.exists(path):
        path = "data/raw/transactions_fraud.csv"
    df = pd.read_csv(path)
    if len(df) > n:
        df = df.sample(n, random_state=42)
    return df


# Sidebar
with st.sidebar:
    st.title("PaySphere Risk Engine")

    section = st.radio(
        "Navigation",
        ["Overview", "Live Scoring", "Analytics", "Pipeline"],
    )

    st.markdown("---")

    threshold = st.slider(
        "Fraud Decision Threshold",
        min_value=0.001,
        max_value=0.5,
        value=0.08333,
        step=0.001,
        format="%.5f",
    )

    show_raw = st.checkbox("Show raw data sample", False)

scorer = load_scorer(threshold)

st.markdown("# PaySphere Fraud Detection System")
st.markdown(
    "Production-grade ML risk engine for digital payment fraud prevention."
)

st.write("")

if section == "Overview":
    render_overview(load_sample_data)

elif section == "Live Scoring":
    render_live_scoring(scorer, threshold)

elif section == "Analytics":
    render_analytics(load_sample_data, show_raw, threshold, scorer)

elif section == "Pipeline":
    render_pipeline()