import time
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from app.ui_components import info_card, chart_card, render_threshold_explanation

def render_live_scoring(scorer, threshold: float):
    # Header with Interceptor Identity
    col_t, col_s = st.columns([3, 1])
    with col_t:
        st.title("⚡ Live Interceptor Terminal")
        st.caption("Manual transaction override and real-time risk diagnostic suite.")
    
    with col_s:
        # API Handshake Pulse
        st.markdown("""
            <div style="background: rgba(76, 139, 245, 0.1); border: 1px solid #4C8BF5; 
                        padding: 10px; border-radius: 8px; text-align: center;">
                <span style="color: #4C8BF5; font-weight: 700;">PROXIED TO RENDER</span>
            </div>
        """, unsafe_allow_html=True)

    render_threshold_explanation(threshold)
    st.write("---")

    # Layout: Form on Left, Diagnostic on Right
    col_form, col_diag = st.columns([1.2, 1.8])

    with col_form:
        with st.form("txn_form", border=True):
            st.subheader("Transaction Parameters")
            amount = st.number_input("Amount (₹)", 1.0, value=12500.0)
            payment_method = st.selectbox("Payment Method", ["UPI", "CARD", "NETBANKING", "WALLET"])
            merchant_category = st.selectbox("Merchant Category", ["Travel", "Electronics", "Fashion", "Gaming"])
            
            st.divider()
            ip_risk = st.slider("IP Reputation Risk", 0.0, 1.0, 0.85)
            device_trust = st.slider("Device Trust Score", 0.0, 1.0, 0.20)
            txn_count = st.number_input("Txn Count (24h)", 0, value=5)

            submitted = st.form_submit_button("⚡ EXECUTE RISK ANALYSIS", use_container_width=True, type="primary")

    if submitted:
        # 1. API Call Logic (Simplified for brevity)
        df_input = pd.DataFrame([{
            "transaction_id": "TXN_LIVE", "amount": amount, "payment_method": payment_method.upper(),
            "merchant_category": merchant_category, "ip_address_risk_score": ip_risk,
            "device_trust_score": device_trust, "txn_count_last_24h": txn_count,
            "threshold": threshold, # Sending UI threshold to API
            # ... include other required fields from your sanitizer
        }])

        label, action, prob = scorer.predict_label_and_action(df_input)

        with col_diag:
            # Result Banner
            banner_color = "#ef4444" if label == 1 else "#10b981"
            st.markdown(f"""
                <div style="background: {banner_color}22; border: 1px solid {banner_color}; 
                            padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                    <h3 style="color: {banner_color}; margin: 0;">DECISION: {action}</h3>
                    <p style="margin: 0; color: #CBD5E1;">Probability: {prob:.2%}</p>
                </div>
            """, unsafe_allow_html=True)

            # Diagnostic Gauge wrapped in chart_card
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=prob*100,
                gauge={'bar': {'color': banner_color}, 'axis': {'range': [0, 100]},
                       'threshold': {'line': {'color': "white", 'width': 2}, 'value': threshold*100}}
            ))
            chart_card("Risk Diagnostic", "Real-time probability score from Random Forest Engine.", fig_g, height=250)

    # Technical Depth: Raw Payload Preview
    with st.expander("🛠️ View API JSON Payload"):
        st.json(df_input.to_dict(orient="records")[0])