import os
import sys
import pandas as pd
import streamlit as st
import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.overview_view import render_overview
from app.live_view import render_live_scoring
from app.analytics_view import render_analytics
from app.pipeline_view import render_pipeline

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

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
.stApp { background: #0f1115; color: #e5e7eb; }
section[data-testid="stSidebar"] { background: #151821; padding-top: 1.5rem; }
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
button:focus { outline: none; box-shadow: 0 0 0 2px var(--accent-soft); }
</style>
""", unsafe_allow_html=True)

# =========================
# API PROXY CLIENT
# =========================
class APIFraudScorer:
    """
    Acts as a proxy for the ML model. The Streamlit views think they are talking 
    to a local model, but this actually routes the data to our FastAPI backend.
    """
    def __init__(self, api_url, threshold=0.5):
        self.api_url = api_url
        self.threshold = threshold

    def predict_label_and_action(self, df_txn: pd.DataFrame):
        try:
            # 1. Convert the dataframe row to a JSON payload
            payload = df_txn.iloc[0].fillna(0).to_dict()
            
            # 2. Ping the FastAPI endpoint
            response = requests.post(f"{self.api_url}/v1/score", json=payload, timeout=5)
            response.raise_for_status() # Raise error for bad HTTP status codes
            
            data = response.json()
            prob = data["fraud_probability"]
            
            # 3. Apply the UI-controlled threshold logic locally
            label = int(prob >= self.threshold)
            if prob >= min(self.threshold * 5, 0.85):
                action = "HARD_BLOCK"
            elif prob >= (self.threshold * 3):
                action = "MANUAL_REVIEW"
            elif prob >= self.threshold:
                action = "OTP_VERIFICATION"
            else:
                action = "ALLOW"
                
            return label, action, prob
            
        except requests.exceptions.RequestException as e:
            st.error(f"API Connection Error. Is FastAPI running? Details: {e}")
            return 0, "ERROR", 0.0
            
    def predict_proba(self, df_txn: pd.DataFrame):
        """Fallback for batch scoring in the analytics view."""
        # For true enterprise analytics, this should hit a batch API endpoint.
        # For now, we mock the return to prevent the view from breaking.
        _, _, prob = self.predict_label_and_action(df_txn)
        return prob
    
    def predict_proba_batch(self, df_batch: pd.DataFrame):
        """Sends a dataframe as a single JSON array to the batch endpoint."""
        try:
            # Convert dataframe to a list of dictionaries
            payload = df_batch.fillna(0).to_dict(orient="records")
            
            # Make ONE request with a slightly longer timeout for the batch computation
            response = requests.post(f"{self.api_url}/v1/batch-score", json=payload, timeout=15)
            response.raise_for_status()
            
            return response.json()["probabilities"]
        except requests.exceptions.RequestException as e:
            st.error(f"Batch API Error: {e}")
            return [0.0] * len(df_batch) # Fallback to prevent UI crash

# =========================
# LOADERS
# =========================
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
    st.html(
        "<div style='display:flex; align-items:center; gap:10px; font-size:18px; "
        "font-weight:700; margin-bottom:20px; color:#e5e7eb;'>"
        "🛡️ PaySphere Risk Engine"
        "</div>"
    )

    # API Health Check Indicator
    try:
        # Use the dynamic API_URL and increase timeout to 5s for Render cold starts
        health = requests.get(f"{API_URL}/health", timeout=5)
        api_status = "🟢 Active & Scoring" if health.status_code == 200 else "🔴 API Error"
        bg_color, border_color, text_color = ("#064e3b", "#047857", "#34d399") if health.status_code == 200 else ("#7f1d1d", "#991b1b", "#fca5a5")
    except Exception as e:
        api_status = "🔴 API Offline"
        bg_color, border_color, text_color = ("#7f1d1d", "#991b1b", "#fca5a5")

    st.html(
        f"<div style='background-color: {bg_color}; color: {text_color}; padding: 8px 12px; "
        f"border-radius: 6px; font-size: 0.85rem; margin-bottom: 24px; border: 1px solid {border_color};'>"
        f"<b>System Status:</b> {api_status}"
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
# Initialize our new API proxy instead of the heavy local model
scorer = APIFraudScorer(api_url=API_URL, threshold=threshold)

if st.session_state.section == "Overview":
    render_overview(load_sample_data, scorer)
elif st.session_state.section == "Live Scoring":
    render_live_scoring(scorer, threshold)
elif st.session_state.section == "Analytics":
    render_analytics(load_sample_data, show_raw, threshold, scorer)
elif st.session_state.section == "Pipeline":
    render_pipeline()