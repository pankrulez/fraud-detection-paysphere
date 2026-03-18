import os
import sys
import pandas as pd
import streamlit as st
import requests
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.overview_view import render_overview
from app.live_view import render_live_scoring
from app.analytics_view import render_analytics
from app.pipeline_view import render_pipeline 
from app.batch_view import render_batch_processing

API_URL = st.secrets.get("API_URL", "http://localhost:8000")

def check_api_health(self):
    try:
        # Increase timeout for the health check specifically to 10s 
        # to allow Render to wake up from its sleep state.
        url = f"{self.api_url.rstrip('/')}/health"
        response = requests.get(url, timeout=10) 
        return response.status_code == 200
    except:
        return False

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
            payload["threshold"] = self.threshold
            
            # 2. Ping the FastAPI endpoint
            response = requests.post(f"{self.api_url}/v1/score", json=payload, timeout=5)
            # Raise error for bad HTTP status codes
            response.raise_for_status() 
            
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
        
        _, _, prob = self.predict_label_and_action(df_txn)
        return prob
    
    def predict_proba_batch(self, df_batch: pd.DataFrame, chunk_size: int = 5000):
        all_probs = []
        
        # 1. Define the EXACT fields expected by your TransactionRequest in api.py
        # Any field NOT in this list will be stripped out before sending to the API.
        api_expected_fields = [
            "customer_id", 
            "device_id", 
            "merchant_id", 
            "timestamp", 
            "amount",
            "payment_method", 
            "is_international", 
            "merchant_category",
            "ip_address_risk_score", 
            "device_trust_score", 
            "txn_count_last_24h",
            "location_change_flag", 
            "otp_success_rate_customer", 
            "past_fraud_count_customer",
            "past_disputes_customer", 
            "merchant_historical_fraud_rate", 
            "hour_of_day",
            "day_of_week", 
            "is_weekend", 
            "customer_tenure_days", 
            "ip_address_country_match",
            "threshold" 
        ]

        # 2. Sanitize the records
        records = []
        df_clean = df_batch.fillna(0) # Handle any NaNs
        
        for _, row in df_clean.iterrows():
            # Convert row to dict
            raw_record = row.to_dict()
            
            # Inject the current threshold from the slider
            raw_record["threshold"] = float(self.threshold)
            
            # Strict Type Casting & Sanitization
            sanitized = {}
            for field in api_expected_fields:
                val = raw_record.get(field)
                
                # Special handling for ID fields (CSV has them as int, API wants str)
                if field in ["customer_id", "device_id", "merchant_id"]:
                    sanitized[field] = str(val)
                # Ensure float fields are floats
                elif field in ["amount", "ip_address_risk_score", "device_trust_score", 
                            "otp_success_rate_customer", "merchant_historical_fraud_rate", "threshold"]:
                    sanitized[field] = float(val) if val is not None else 0.0
                # Ensure int fields are ints
                elif field in ["is_international", "is_weekend", "txn_count_last_24h", 
                            "hour_of_day", "day_of_week", "location_change_flag"]:
                    sanitized[field] = int(val) if val is not None else 0
                else:
                    sanitized[field] = val
            
            records.append(sanitized)

        # 3. Chunked Transmission
        chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]
        
        try:
            for chunk in chunks:
                response = requests.post(
                    f"{self.api_url}/v1/batch-score", 
                    json=chunk, 
                    timeout=120
                )
                
                if response.status_code == 422:
                    # BREAK THE LOOP: If the first chunk fails, don't keep trying.
                    error_detail = response.json()
                    st.error(f"❌ API Validation Error: {error_detail}")
                    st.stop() # Stops Streamlit execution here
                
                response.raise_for_status()
                all_probs.extend(response.json()["probabilities"])
                
            return all_probs
            
        except Exception as e:
            st.error(f"⚠️ Batch Processing Stopped: {str(e)}")
            st.stop()
        
    def check_api_health(self):
        """Checks if the FastAPI backend is responding via the /health endpoint."""
        try:
            # Explicitly target the /health endpoint
            url = f"{self.api_url.rstrip('/')}/health"
            response = requests.get(url, timeout=5)
            
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

# Page Configuration
st.set_page_config(
    page_title="PaySphere Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# SESSION STATE INITIALIZATION
if 'api_scorer' not in st.session_state:
    st.session_state.api_scorer = APIFraudScorer(api_url=API_URL)

# =========================
# CLEAN ENTERPRISE STYLING
# =========================
st.markdown("""
<style>
:root {
    --accent: #4C8BF5;
    --accent-soft: #1e3a8a;
}
.stApp { 
    background: #0f1115; 
    color: #e5e7eb; 
}
section[data-testid="stSidebar"] { 
    background: #151821; 
    padding-top: 1.5rem; 
}
div[data-testid="stButton"] > button {
    background-color: transparent; 
    color: #cbd5e1;
    border: 1px solid transparent; 
    border-radius: 8px;
    height: 42px; 
    text-align: left; 
    font-weight: 500;
    transition: all 0.2s ease-in-out; 
    width: 100%;
}
div[data-testid="stButton"] > button:hover {
    background-color: #1f2937; 
    border-color: #334155; 
    color: #ffffff;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background-color: #1f2937 !important; 
    border-left: 3px solid var(--accent) !important;
    color: #ffffff !important; 
    font-weight: 600 !important;
}
button:focus { 
    outline: none; 
    box-shadow: 0 0 0 2px var(--accent-soft); 
}
</style>
""", unsafe_allow_html=True)



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
    nav_button("Batch Processing", "📂", "nav_batch")
    nav_button("Analytics", "📊", "nav_analytics")
    nav_button("Pipeline", "⚙️", "nav_pipeline")

    st.write("---")

    st.caption("RISK CONTROLS")
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.001, 
        max_value=0.5, 
        value=0.083, 
        step=0.001, 
        format="%.3f",
        help="Lower threshold = Higher Recall (Catches more fraud, more false alarms). Higher threshold = Higher Precision (Fewer false alarms)."
    )

    if threshold < 0.03:
        st.info("Mode: Aggressive Capture", icon="🛡️")
    elif threshold < 0.15:
        st.success("Mode: Balanced", icon="⚖️")
    else:
        st.warning("Mode: High Precision", icon="🎯")

    show_raw = st.checkbox("Show Raw Data in Analytics", False)

# ==========
# ROUTING
# ==========
# Initialize API proxy
scorer = APIFraudScorer(api_url=API_URL, threshold=threshold)

if st.session_state.section == "Overview":
    render_overview(load_sample_data, scorer)
elif st.session_state.section == "Live Scoring":
    render_live_scoring(scorer, threshold)
elif st.session_state.section == "Batch Processing":
    render_batch_processing(scorer)
elif st.session_state.section == "Analytics":
    render_analytics(load_sample_data, show_raw, threshold, scorer)
elif st.session_state.section == "Pipeline":
    render_pipeline()