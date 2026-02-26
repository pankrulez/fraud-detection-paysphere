# app/app.py
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

# ---------- GLOBAL UI ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #0f172a, #111827, #1e293b, #0f172a);
    background-size: 400% 400%;
    animation: gradientBG 18s ease infinite;
    color: #e5e7eb;
}
@keyframes gradientBG {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}
.glass-card {
    background: rgba(17,24,39,0.75);
    backdrop-filter: blur(14px);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    animation: fadeIn 0.6s ease-in-out;
}
@keyframes fadeIn {
    from {opacity:0; transform: translateY(8px);}
    to {opacity:1; transform: translateY(0);}
}
section[data-testid="stSidebar"] {
    background: #0b1220;
}
[data-testid="stMetric"] {
    background: rgba(30,41,59,0.6);
    padding: 14px;
    border-radius: 12px;
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
        ["🏠 Overview", "🔍 Live Scoring", "📊 Analytics & Plots", "🧬 Project Pipeline"],
    )

    st.markdown("---")
    threshold = st.slider(
    "Fraud cutoff (decision threshold)",
    min_value=0.001,
    max_value=0.5,
    value=0.08333,
    step=0.001,
    format="%.5f",
)

    show_raw = st.checkbox("Show raw data sample in Analytics", False)

scorer = load_scorer(threshold)


# Header
st.markdown("# PaySphere Fraud Detection System")
st.markdown("End-to-end ML-powered risk engine for UPI, cards, net banking, and wallets.")


st.write("")

if section == "🏠 Overview":
    render_overview(load_sample_data)

elif section == "🔍 Live Scoring":
    render_live_scoring(scorer, threshold)

elif section == "📊 Analytics & Plots":
    render_analytics(load_sample_data, show_raw, threshold, scorer)

elif section == "🧬 Project Pipeline":
    render_pipeline()