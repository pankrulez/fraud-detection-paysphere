import os
import sys
import pandas as pd
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.modeling.inference import FraudScorer
from app.overview_view import render_overview
from app.live_view import render_live_scoring
from app.analytics_view import render_analytics
from app.pipeline_view import render_pipeline

st.set_page_config(
    page_title="PaySphere Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# CLEAN ENTERPRISE STYLING
# =========================
st.markdown("""
<style>
:root {
    --accent: #4C8BF5;
    --accent-soft: #1e3a8a;
}
/* Neutral dark background */
.stApp { background: #0f1115; color: #e5e7eb; }
/* Sidebar neutral graphite */
section[data-testid="stSidebar"] { background: #151821; padding-top: 1.5rem; }

/* Custom Navigation Buttons */
div[data-testid="stButton"] > button {
    background-color: transparent; color: #cbd5e1;
    border: 1px solid transparent; border-radius: 8px;
    height: 42px; text-align: left; font-weight: 500;
    transition: all 0.2s ease-in-out; width: 100%;
}
div[data-testid="stButton"] > button:hover {
    background-color: #1f2937; border-color: #334155; color: #ffffff;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background-color: #1f2937 !important; border-left: 3px solid var(--accent) !important;
    color: #ffffff !important; font-weight: 600 !important;
}
/* Focus ring */
button:focus { outline: none; box-shadow: 0 0 0 2px var(--accent-soft); }
</style>
""", unsafe_allow_html=True)

# =========================
# LOADERS
# =========================
@st.cache_resource(show_spinner="Loading Model Artifacts...")
def load_scorer(threshold: float = 0.5):
    return FraudScorer(
        model_path="models/artifacts/fraud_model.joblib",
        encoders_path="models/encoders/preprocessing.joblib",
        threshold=threshold,
    )

@st.cache_data(show_spinner="Loading Transaction Data...")
def load_sample_data(n: int = 50000):
    path = "data/interim/transactions_clean.csv"
    if not os.path.exists(path):
        path = "data/raw/transactions_fraud.csv"
    df = pd.read_csv(path)
    if len(df) > n:
        df = df.sample(n, random_state=42)
    return df

# =========================
# SIDEBAR NAVIGATION
# =========================
with st.sidebar:
    
    # 1. Header (Using st.html to bypass Markdown parser)
    st.html(
        "<div style='display:flex; align-items:center; gap:10px; font-size:18px; "
        "font-weight:700; margin-bottom:20px; color:#e5e7eb;'>"
        "🛡️ PaySphere Risk Engine"
        "</div>"
    )

    # 2. System Status Indicator (Flattened string)
    st.html(
        "<div style='background-color: #064e3b; color: #34d399; padding: 8px 12px; "
        "border-radius: 6px; font-size: 0.85rem; margin-bottom: 24px; border: 1px solid #047857;'>"
        "🟢 <b>System Status:</b> Active & Scoring"
        "</div>"
    )

    if "section" not in st.session_state:
        st.session_state.section = "Overview"

    def set_section(name):
        st.session_state.section = name

    def nav_button(label, icon, key):
        is_active = st.session_state.section == label
        st.button(
            f"{icon} {label}", key=key,
            type="primary" if is_active else "secondary",
            on_click=set_section, args=(label,),
            use_container_width=True
        )

    st.caption("NAVIGATION")
    nav_button("Overview", "📋", "nav_overview")
    nav_button("Live Scoring", "⚡", "nav_live")
    nav_button("Analytics", "📊", "nav_analytics")
    nav_button("Pipeline", "⚙️", "nav_pipeline")

    st.write("---")

    # Risk Controls
    st.caption("RISK CONTROLS")
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.001, max_value=0.5, value=0.083, step=0.001, format="%.3f",
        help="Lower threshold = Higher Recall (Catches more fraud, more false alarms). Higher threshold = Higher Precision (Fewer false alarms)."
    )

    if threshold < 0.03:
        st.info("Mode: Aggressive Capture", icon="🛡️")
    elif threshold < 0.15:
        st.success("Mode: Balanced", icon="⚖️")
    else:
        st.warning("Mode: High Precision", icon="🎯")

    show_raw = st.checkbox("Show Raw Data in Analytics", False)

# =========================
# MAIN CONTENT ROUTING
# =========================
scorer = load_scorer(threshold)

if st.session_state.section == "Overview":
    render_overview(load_sample_data)
elif st.session_state.section == "Live Scoring":
    render_live_scoring(scorer, threshold)
elif st.session_state.section == "Analytics":
    render_analytics(load_sample_data, show_raw, threshold, scorer)
elif st.session_state.section == "Pipeline":
    render_pipeline()