import os
import streamlit as st
from app.ui_components import info_card

def render_pipeline():
    # Force a check every time the tab is clicked
    scorer = st.session_state.get('api_scorer')
    
    # show a small spinner to be sure it's running
    if scorer:
        is_online = scorer.check_api_health()
    else:
        is_online = False
    
    # 1. UI STYLING
    st.markdown("""
        <style>
        .pipeline-card {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid #334155;
            margin-bottom: 20px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        .step-number {
            background: #3B82F6;
            color: white;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 12px;
        }
        .registry-container {
            background: #111827;
            border-radius: 12px;
            border-left: 4px solid #4F46E5;
            padding: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ Fraud Detection Pipeline")
    st.caption("Enterprise-grade ML architecture for high-frequency payment streams.")

    st.write(
        """
        The PaySphere pipeline transforms raw transaction logs into calibrated risk decisions. 
        By utilizing a **decoupled microservice architecture**, we separate the inference logic 
        from the presentation layer to ensure maximum scalability and reliability.
        """
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Content steps for the UI
    steps = [
        {
            "step": 1, "title": "Data Ingestion & Contract", "accent": "#3B82F6",
            "obj": "Strict schema validation via Pydantic to ensure data integrity.",
            "meth": ["Implemented FastAPI Request Models", "Validated 20+ feature fields", "Automatic Type Enforcement"],
            "out": "Validated JSON payload ready for transformation."
        },
        {
            "step": 2, "title": "Behavioral Feature Engineering", "accent": "#10B981",
            "obj": "Extract high-signal fraud indicators without data leakage.",
            "meth": ["Time-series expanding windows", "Transaction velocity metrics", "Merchant risk profiles"],
            "out": "Engineered feature matrix capturing behavioral risk signals."
        },
        {
            "step": 3, "title": "Unified Inference Pipeline", "accent": "#F59E0B",
            "obj": "Handle scaling, encoding, and modeling in one atomic operation.",
            "meth": ["StandardScaler normalization", "One-Hot Categorical Encoding", "RandomForest Classification"],
            "out": "Single joblib artifact optimized for sub-50ms inference."
        },
        {
            "step": 4, "title": "Real-Time Scoring API", "accent": "#EC4899",
            "obj": "Serve predictions via RESTful endpoints.",
            "meth": ["Vectorized Batch Scoring", "Single Transaction Latency Tuning", "Cross-Origin Support"],
            "out": "Live HTTP scoring service running on Render."
        }
    ]

    for s in steps:
        with st.container():
            st.markdown(f"""
                <div class="pipeline-card">
                    <div style="display: flex; align-items: center; margin-bottom: 12px;">
                        <span class="step-number" style="background-color: {s['accent']};">{s['step']}</span>
                        <h3 style="margin: 0; color: white;">{s['title']}</h3>
                    </div>
                    <p style="color: #94A3B8; font-size: 0.95rem;"><b>Objective:</b> {s['obj']}</p>
                    <ul style="color: #CBD5E1; font-size: 0.9rem;">
                        {"".join([f"<li>{m}</li>" for m in s['meth']])}
                    </ul>
                    <div style="background: rgba(16, 185, 129, 0.1); padding: 8px 12px; border-radius: 6px; border-left: 3px solid #10B981;">
                        <span style="color: #10B981; font-weight: 600; font-size: 0.85rem;">OUTPUT:</span>
                        <span style="color: #E5E7EB; font-size: 0.85rem;"> {s['out']}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # 2. ENHANCED MODEL REGISTRY (API BASED)
    st.markdown("---")
    st.subheader("📦 Production Model Registry")
    
    # Get the scorer from session state
    scorer = st.session_state.get('api_scorer')
    is_online = scorer.check_api_health() if scorer else False
    
    status_color = "#10B981" if is_online else "#EF4444"
    status_text = "ONLINE" if is_online else "OFFLINE"

    st.markdown(f"""
        <div class="registry-container">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p style="margin:0; color:#64748B; font-size:0.8rem; text-transform: uppercase;">Registry Artifact</p>
                    <p style="margin:0; font-weight:600; color:#F8FAFC;">fraud_pipeline.joblib</p>
                </div>
                <div>
                    <p style="margin:0; color:#64748B; font-size:0.8rem; text-transform: uppercase;">Backend Status</p>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="height: 10px; width: 10px; background-color: {status_color}; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px {status_color};"></span>
                        <p style="margin:0; font-weight:600; color:{status_color};">{status_text}</p>
                    </div>
                </div>
                <div>
                    <p style="margin:0; color:#64748B; font-size:0.8rem; text-transform: uppercase;">Environment</p>
                    <p style="margin:0; font-weight:600; color:#F59E0B;">Render Cloud (Python 3.11)</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)