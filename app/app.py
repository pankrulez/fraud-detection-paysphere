import os
import sys
import pandas as pd
import streamlit as st
import plotly.io as pio
import plotly.graph_objects as go

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.modeling.inference import FraudScorer
from app.overview_view import render_overview
from app.live_view import render_live_scoring
from app.analytics_view import render_analytics
from app.pipeline_view import render_pipeline

pio.templates["neutral_dark"] = go.layout.Template(
    layout=dict(
        paper_bgcolor="#0f1115",
        plot_bgcolor="#0f1115",
        font=dict(color="#e5e7eb"),
    )
)

pio.templates.default = "neutral_dark"

st.set_page_config(
    page_title="PaySphere Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# CLEAN ENTERPRISE STYLING
# =========================
st.markdown("""
<style>

:root {
    --accent: #10b981;
    --accent-soft: #064e3b;
}

/* Neutral dark background */
.stApp {
    background: #0f1115;
    color: #e5e7eb;
}

/* Sidebar neutral graphite */
section[data-testid="stSidebar"] {
    background: #151821;
    padding-top: 1.5rem;
}

/* Navigation buttons */
div[data-testid="stButton"] > button {
    background-color: transparent;
    color: #cbd5e1;
    border: 1px solid transparent;
    border-radius: 8px;
    height: 42px;
    text-align: left;
    font-weight: 500;
    transition: all 0.2s ease-in-out;
}

div[data-testid="stButton"] > button:hover {
    background-color: #1f2937;
    border-color: #334155;
    color: #ffffff;
}

/* Active nav */
div[data-testid="stButton"] > button[kind="primary"] {
    background-color: #1f2937 !important;
    border-left: 3px solid var(--accent) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Slider accent */
div[data-baseweb="slider"] div[role="slider"] {
    background-color: var(--accent) !important;
}

/* KPI cards */
[data-testid="stMetric"] {
    background: #181c24;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.06);
}

/* Focus ring */
button:focus {
    outline: none;
    box-shadow: 0 0 0 2px var(--accent-soft);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOADERS
# =========================
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

# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;
                font-size:16px;font-weight:600;">
        <svg width="16" height="16" viewBox="0 0 24 24"
             fill="none" stroke="#10b981" stroke-width="1.8">
            <path d="M12 2L4 6v6c0 5 3.5 9 8 10 4.5-1 8-5 8-10V6l-8-4z"/>
        </svg>
        PaySphere Risk Intelligence
    </div>
    """, unsafe_allow_html=True)

    st.caption("Fraud Detection Platform")

    st.write("")

    # Navigation
    
    st.subheader("Navigation")

    if "section" not in st.session_state:
        st.session_state.section = "Overview"

    def set_section(name):
        st.session_state.section = name

    def nav_button(label, icon_svg, key):
        is_active = st.session_state.section == label

        st.button(
            f"{icon_svg}  {label}",
            use_container_width=True,
            key=key,
            type="primary" if is_active else "secondary",
            on_click=set_section,
            args=(label,),
        )

    # Minimal SVG icons
    icon_overview = "📋"
    icon_live = "🔎"
    icon_analytics = "📊"
    icon_pipeline = "⚙"

    nav_button("Overview", icon_overview, "nav_overview")
    nav_button("Live Scoring", icon_live, "nav_live")
    nav_button("Analytics", icon_analytics, "nav_analytics")
    nav_button("Pipeline", icon_pipeline, "nav_pipeline")

    section = st.session_state.section
    st.write("")

    # Risk Controls
    st.subheader("Risk Controls")

    threshold = st.slider(
        "Fraud Decision Threshold",
        min_value=0.001,
        max_value=0.5,
        value=0.08333,
        step=0.001,
        format="%.5f",
    )

    # Simple mode indicator
    if threshold < 0.03:
        st.caption("Mode: High Recall (Aggressive Capture)")
    elif threshold < 0.1:
        st.caption("Mode: Balanced")
    else:
        st.caption("Mode: High Precision (Low False Positives)")

    show_raw = st.checkbox("Show Raw Data", False)

# =========================
# MAIN CONTENT
# =========================
scorer = load_scorer(threshold)

if section == "Overview":
    render_overview(load_sample_data)

elif section == "Live Scoring":
    render_live_scoring(scorer, threshold)

elif section == "Analytics":
    render_analytics(load_sample_data, show_raw, threshold, scorer)

elif section == "Pipeline":
    render_pipeline()   