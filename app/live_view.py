import time
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from app.ui_components import info_card, chart_card, render_threshold_explanation

def render_live_scoring(scorer, threshold: float):
    is_online = scorer.check_api_health() if scorer else False
    status_color = "#10B981" if is_online else "#EF4444"
    status_text = "TERMINAL ACTIVE" if is_online else "TERMINAL OFFLINE"
    sub_text = "Direct API Bridge" if is_online else "No Connection"
    
    # 1. HEADER SECTION (Command Center Style)
    col_t, col_s = st.columns([3, 1])
    with col_t:
        st.title("⚡ Live Interceptor Terminal")
        st.caption("Real-time behavioral diagnostic suite proxiied to PaySphere Risk API.")
    
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

    # 2. INPUT & DIAGNOSTIC LAYOUT
    col_form, col_diag = st.columns([1.2, 1.8])

    with col_form:
        with st.form("txn_form", border=True):
            st.subheader("Intercept Parameters")
            amount = st.number_input("Amount (₹)", 1.0, value=12500.0, step=500.0)
            payment_method = st.selectbox("Payment Rail", ["UPI", "CARD", "NETBANKING", "WALLET"])
            merchant_category = st.selectbox("Merchant Category", ["Travel", "Electronics", "Fashion", "Gaming", "Utilities"])
            
            st.divider()
            ip_risk = st.slider("IP Reputation Risk", 0.0, 1.0, 0.85)
            device_trust = st.slider("Device Trust Score", 0.0, 1.0, 0.20)
            txn_count = st.number_input("Txn Count (24h)", 0, value=5)
            is_intl = st.toggle("International Transaction", value=False)

            submitted = st.form_submit_button("⚡ EXECUTE RISK ANALYSIS", use_container_width=True, type="primary")

    if submitted:
        # DATA PREPARATION
        now = datetime.now()
        df_input = pd.DataFrame([{
            "customer_id": "C_LIVE_USR", "device_id": "D_LIVE_DEV", "merchant_id": "M_LIVE_MERCH",
            "timestamp": now.isoformat(), "amount": float(amount),
            "payment_method": payment_method.lower(), "merchant_category": merchant_category.lower(),
            "ip_address_risk_score": float(ip_risk), "device_trust_score": float(device_trust),
            "is_international": int(is_intl), "is_weekend": 1 if now.weekday() >= 5 else 0,
            "past_fraud_count_customer": 0, "past_disputes_customer": 0,
            "txn_count_last_24h": int(txn_count), "customer_tenure_days": 365,
            "location_change_flag": 1 if ip_risk > 0.7 else 0,
            "otp_success_rate_customer": 0.9, "ip_address_country_match": 1,
            "hour_of_day": now.hour, "day_of_week": now.weekday(),
            "merchant_historical_fraud_rate": 0.02, "threshold": float(threshold)
        }])

        with st.spinner("Analyzing transaction vectors..."):
            label, action, prob = scorer.predict_label_and_action(df_input)

        with col_diag:
            # Result Banner
            status_color = "#ef4444" if label == 1 else "#10b981"
            st.markdown(f"""
                <div style="background: {status_color}11; border: 1px solid {status_color}; 
                            padding: 20px; border-radius: 12px; margin-bottom: 20px; border-left: 8px solid {status_color};">
                    <h2 style="color: {status_color}; margin: 0;">{action}</h2>
                    <p style="margin: 5px 0 0 0; color: #94a3b8; font-size: 1.1rem;">Risk Probability: <b>{prob:.2%}</b></p>
                </div>
            """, unsafe_allow_html=True)

            # A: Styled Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=prob*100,
                number={'suffix': "%", 'font': {'color': '#F8FAFC'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#334155"},
                    'bar': {'color': status_color},
                    'bgcolor': "rgba(0,0,0,0)",
                    'threshold': {'line': {'color': "white", 'width': 3}, 'value': threshold*100}
                }
            ))
            chart_card("Risk Probability Gauge", "Current score relative to system threshold.", fig_g, height=250)

    # 3. ADVANCED EXPLAINABILITY SECTION
    if submitted:
        st.write("---")
        st.subheader("🔍 Deep Diagnostic Breakdown")
        
        c1, c2 = st.columns(2)
        
        with c1:
            # NEW PLOT: Radar Analysis
            categories = ['Amount', 'IP Risk', 'Device Trust', 'Velocity', 'Merchant Risk']
            radar_vals = [min(1, amount/50000), ip_risk, 1-device_trust, min(1, txn_count/20), 0.15]
            
            # Convert status_color to RGBA to avoid Plotly validation errors
            # #ef4444 (Red) -> rgba(239, 68, 68, 0.2)
            # #10b981 (Green) -> rgba(16, 185, 129, 0.2)
            rgba_fill = "rgba(239, 68, 68, 0.2)" if label == 1 else "rgba(16, 185, 129, 0.2)"
            
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=radar_vals, 
                theta=categories, 
                fill='toself',
                line_color=status_color, 
                fillcolor=rgba_fill # Using standard RGBA string
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="#334155"), 
                    bgcolor="rgba(0,0,0,0)",
                    angularaxis=dict(gridcolor="#334155", linecolor="#334155")
                ),
                showlegend=False, 
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=40, b=40, l=40, r=40)
            )
            chart_card("Risk Fingerprint", "Transaction profile across key fraud dimensions.", fig_radar, height=350)

        with c2:
            # WATERFALL XAI (Your existing logic, polished)
            safe_baselines = {"ip_address_risk_score": 0.0, "device_trust_score": 1.0, "amount": 500.0, "txn_count_last_24h": 1}
            df_safe = df_input.copy()
            for f, v in safe_baselines.items(): df_safe[f] = v
            base_prob = scorer.predict_proba(df_safe)
            
            impacts = {}
            for f in safe_baselines.keys():
                df_temp = df_safe.copy()
                df_temp[f] = df_input[f].iloc[0]
                impacts[f] = scorer.predict_proba(df_temp) - base_prob
            
            sorted_impacts = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)
            x_vals = [base_prob] + [i for _, i in sorted_impacts]
            y_labs = ["Baseline"] + [f.replace("_", " ").title()[:15] for f, _ in sorted_impacts]
            
            fig_wat = go.Figure(go.Waterfall(
                orientation="h", measure=["absolute"] + ["relative"]*len(sorted_impacts),
                y=y_labs, x=x_vals,
                decreasing={"marker": {"color": "#10b981"}}, increasing={"marker": {"color": "#ef4444"}}
            ))
            fig_wat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"))
            chart_card("Contribution Analysis", "Mathematical weight of each feature on the final score.", fig_wat, height=350)

        # Bottom Row: Technical Audit
        with st.expander("🛠️ View API Transaction Payload"):
            st.json(df_input.to_dict(orient="records")[0])