import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
from app.ui_components import chart_card, info_card, render_threshold_explanation

def render_analytics(load_sample_data_fn, show_raw: bool, threshold: float, scorer):
    is_online = scorer.check_api_health() if scorer else False
    status_color = "#10B981" if is_online else "#EF4444"
    status_text = "SIMULATION ACTIVE" if is_online else "SIMULATION OFFLINE"
    sub_text = "Batch Vectorized" if is_online else "API Unreachable"
    
    # 1. HEADER & SIMULATOR STATUS
    col_t, col_s = st.columns([3, 1])
    with col_t:
        st.title("📊 Intelligence & ROI Simulator")
        st.caption("Strategic impact analysis and financial risk modeling across 25,000 transaction samples.")
    
    with col_s:
        st.markdown(f"""
            <div style="background: {status_color}11; border: 1px solid {status_color}; 
                        padding: 12px; border-radius: 10px; text-align: center;">
                <span style="color: {status_color}; font-weight: 700; font-size: 0.85rem;">● {status_text}</span><br>
                <span style="color: #94a3b8; font-size: 0.7rem;">{sub_text}</span>
            </div>
        """, unsafe_allow_html=True)

    render_threshold_explanation(threshold)
    st.write("---")

    # 2. DATA SCORING ENGINE (Optimized for 25k samples)
    if 'analytics_data' not in st.session_state:
        with st.spinner("Executing batch inference for 25k samples..."):
            df_raw = load_sample_data_fn().head(25000)
            
            # --- SCHEMA ALIGNMENT (Ensuring API Compliance) ---
            defaults = {
                "customer_id": "C_ANALYSIS", "device_id": "D_ANALYSIS", "merchant_id": "M_ANALYSIS",
                "timestamp": datetime.utcnow().isoformat(), "is_international": 0, "is_weekend": 0,
                "past_fraud_count_customer": 0, "past_disputes_customer": 0, "txn_count_last_24h": 0,
                "customer_tenure_days": 365, "location_change_flag": 0, "otp_success_rate_customer": 1.0,
                "ip_address_country_match": 1, "hour_of_day": 12, "day_of_week": 3,
                "merchant_historical_fraud_rate": 0.05, "ip_address_risk_score": 0.1,
                "device_trust_score": 0.9, "amount": 100.0, "payment_method": "upi",
                "merchant_category": "retail"
            }
            for col, val in defaults.items():
                if col not in df_raw.columns:
                    df_raw[col] = val
            
            # Vectorized API Call
            df_raw['fraud_prob'] = scorer.predict_proba_batch(df_raw)
            st.session_state.analytics_data = df_raw

    df = st.session_state.analytics_data.copy()
    df['is_blocked'] = (df['fraud_prob'] >= threshold).astype(int)

    # 3. FINANCIAL IMPACT STRIP
    # Logic: Prevented Loss (TP), Leakage (FN), Friction (FP)
    saved = df[(df['is_blocked'] == 1) & (df['is_fraud'] == 1)]['amount'].sum()
    leakage = df[(df['is_blocked'] == 0) & (df['is_fraud'] == 1)]['amount'].sum()
    friction = df[(df['is_blocked'] == 1) & (df['is_fraud'] == 0)]['amount'].sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fraud Loss Prevented", f"₹{saved:,.0f}", help="Total capital loss averted.")
    m2.metric("Revenue Leakage", f"₹{leakage:,.0f}", delta="Missed Fraud", delta_color="inverse")
    m3.metric("Friction Value", f"₹{friction:,.0f}", delta="False Alarms", delta_color="inverse")
    m4.metric("Capture Efficiency", f"{(saved / (saved + leakage + 1e-6)):.1%}")

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. STRATEGIC VISUALIZATIONS (Turbo High-Contrast)
    col_left, col_right = st.columns(2)

    with col_left:
        # TREEMAP: Where is the model concentrating its power?
        fig_tree = px.treemap(
            df[df['fraud_prob'] > threshold], 
            path=['merchant_category', 'payment_method'], 
            values='amount', color='fraud_prob',
            color_continuous_scale='Turbo',
            title="Density of Blocked Capital"
        )
        fig_tree.update_layout(margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)")
        chart_card("Risk Concentration", "Identifying sectors where the model is actively intercepting high-value fraud.", fig_tree)

    with col_right:
        # PROBABILITY DISTRIBUTION: Visualizing the Decision Boundary
        fig_dist = px.histogram(
            df, x="fraud_prob", nbins=50, 
            color_discrete_sequence=["#3B82F6"],
            title="Population Risk Distribution"
        )
        fig_dist.add_vline(x=threshold, line_dash="dash", line_color="#ef4444", annotation_text="Active Threshold")
        fig_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Fraud Probability")
        chart_card("Model Decision Cut-off", "Distribution of scores relative to the current decision threshold.", fig_dist)

    st.write("---")

    # 5. STRATEGIC INTELLIGENCE SUMMARY (Clean HTML)
    st.subheader("Intelligence Report")
    
    report_html = f"""
        <p style='color: #CBD5E1; font-size: 0.9rem; line-height: 1.6; margin: 0;'>
            • <b>Friction Ratio:</b> For every ₹1 of fraud blocked, the system creates ₹{(friction/(saved+1e-6)):.2f} in potential customer friction.<br>
            • <b>Model Sharpness:</b> The top 5% of highest-risk transactions account for <b>{(df.sort_values('fraud_prob', ascending=False).head(int(len(df)*0.05))['amount'].sum() / df['amount'].sum()):.1%}</b> of total transaction value.<br>
            • <b>Impact Score:</b> Current settings are preventing <b>{(saved/(saved+leakage+1e-6)):.1%}</b> of all possible capital loss in this dataset.
        </p>
    """.strip()
    
    info_card("Strategic Intelligence Insights", report_html, accent="warning")

    if show_raw:
        st.write("---")
        st.subheader("🚩 High-Risk Transaction Manifest (Simulated)")
        st.dataframe(df[df['is_blocked'] == 1].sort_values('fraud_prob', ascending=False), use_container_width=True)