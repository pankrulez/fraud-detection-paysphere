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

# ==========================================
# SIDEBAR: RISK INTELLIGENCE CONSOLE
# ==========================================
with st.sidebar:
    # 1. BRANDING HEADER
    st.markdown("""
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <h1 style="color: #4C8BF5; margin-bottom: 0; font-size: 1.8rem;">PaySphere</h1>
            <p style="color: #94A3B8; font-size: 0.8rem; margin-top: 0; letter-spacing: 1px;">RISK INTELLIGENCE v1.0</p>
        </div>
    """, unsafe_allow_html=True)

    # 2. REAL-TIME API HEARTBEAT
    try:
        # Pinging the Render FastAPI backend
        health = requests.get(f"{API_URL}/health", timeout=5)
        is_active = health.status_code == 200
        status_text = "SYSTEM OPERATIONAL" if is_active else "LATENCY DETECTED"
        pulse_color = "#10B981" if is_active else "#F59E0B"
    except Exception:
        is_active = False
        status_text = "ENGINE OFFLINE"
        pulse_color = "#EF4444"

    st.markdown(f"""
        <div style="background: {pulse_color}11; border: 1px solid {pulse_color}44; 
                    padding: 12px; border-radius: 10px; margin-bottom: 25px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 8px; height: 8px; background: {pulse_color}; border-radius: 50%; box-shadow: 0 0 8px {pulse_color};"></div>
                <span style="color: {pulse_color}; font-weight: 700; font-size: 0.75rem; letter-spacing: 0.5px;">{status_text}</span>
            </div>
            <div style="color: #94a3b8; font-size: 0.65rem; margin-top: 4px; padding-left: 16px;">
                Endpoint: {API_URL.split('//')[-1]}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 3. NAVIGATION LOGIC
    if "section" not in st.session_state:
        st.session_state.section = "Overview"

    def set_section(name):
        st.session_state.section = name

    # Professional Navigation Mapping
    nav_items = [
        {"label": "Overview", "name": "🛡️ Executive Command Center", "icon": "🛡️"},
        {"label": "Live Scoring", "name": "⚡ Real-Time Interceptor", "icon": "⚡"},
        {"label": "Batch Processing", "name": "📂 Bulk Risk Assessment", "icon": "📂"},
        {"label": "Analytics", "name": "📊 Intelligence & ROI Simulator", "icon": "📊"},
        {"label": "Pipeline", "name": "⚙️ MLOps & Model Registry", "icon": "⚙️"}
    ]

    st.caption("OPERATIONAL VIEWS")
    for item in nav_items:
        is_active = st.session_state.section == item["label"]
        st.button(
            item["name"], 
            key=f"nav_{item['label'].lower()}",
            type="primary" if is_active else "secondary",
            on_click=set_section, 
            args=(item["label"],),
            use_container_width=True
        )

    st.write("---")

    # 4. SYSTEM CALIBRATION (RISK CONTROLS)
    st.caption("SYSTEM CALIBRATION")
    threshold = st.slider(
        "Sensitivity Threshold",
        min_value=0.001, 
        max_value=0.500, 
        value=0.083, 
        step=0.001, 
        format="%.3f",
        help="Adjust the decision boundary. Lower values catch more fraud but increase false positives."
    )

    # Dynamic Mode Indicator
    if threshold < 0.03:
        st.error("🛡️ Mode: Aggressive Capture")
    elif threshold < 0.15:
        st.success("⚖️ Mode: Balanced")
    else:
        st.warning("🎯 Mode: High Precision")

    st.write("---")
    show_raw = st.checkbox("🔍 Enable Detailed Manifests", False)

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