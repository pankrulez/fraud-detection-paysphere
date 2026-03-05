import time
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from app.ui_components import chart_card, info_card
from app.ui_components import render_threshold_explanation

def render_live_scoring(scorer, threshold: float):

    st.title("⚡ Real-Time Risk Scoring")
    st.markdown(
        "Simulate incoming transactions and observe the model's probability score, "
        "risk classification, and operational decision in real-time."
    )
    render_threshold_explanation(threshold)
    st.markdown("---")

    # ==========================================
    # TRANSACTION INPUT FORM
    # ==========================================
    with st.form("txn_form", border=True):
        st.subheader("Transaction Intercept Parameters")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            amount = st.number_input(
                "Amount (₹)", 
                1.0, 
                value=12500.0, 
                step=500.0,
                help="Higher amounts relative to the user's history increase the risk weight."
                )
            payment_method = st.selectbox(
                "Payment Method", 
                ["UPI", "CARD", "NETBANKING", "WALLET"]
                )
            merchant_category = st.selectbox(
                "Merchant Category", 
                ["Electronics", "Travel", "Fashion", "Gaming", "Grocery", "Utilities"]
                )

        with col2:
            ip_risk = st.slider(
                "IP Reputation Risk", 
                0.0, 1.0, 0.85, 
                help="1.0 means known proxy/botnet IP"
                )
            device_trust = st.slider(
                "Device Trust Score", 
                0.0, 1.0, 0.20, 
                help="A score based on device fingerprinting. 1.0 is a known, safe device; <0.5 is suspicious."
                )
            txn_count_24h = st.number_input(
                "Txn Count (24h)", 
                0, 
                value=5,
                help="Velocity Check: The number of successful transactions this customer has made in the last 24 hours. Rapid bursts often indicate account takeover."
                )

        with col3:
            merchant_hist_fraud = st.slider(
                "Merchant Fraud Rate", 
                0.0, 1.0, 0.15
                )
            is_international = st.selectbox(
                "Cross-Border (International)?", 
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="Cross-border transactions carry a higher baseline risk coefficient."
                )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡ Execute Real-Time Risk Analysis", type="primary", use_container_width=True)

    # ==========================================
    # SCORING & OUTPUT
    # ==========================================
    if submitted:
        
        # 1. The "Wow" Factor: Simulate heavy processing
        with st.spinner("Intercepting transaction... Routing to FastAPI Risk Engine..."):
            time.sleep(0.5) # Reduced slightly since API call adds real latency

        now = datetime.now()

        # Build Dataframe matching the exact Pydantic schema of our API
        df_input = pd.DataFrame([{
            "transaction_id": "TXN_999999", # Note: API expects strings for IDs now
            "customer_id": "CUST_10234",
            "device_id": "DEV_5555",
            "merchant_id": "MERCH_111",
            "timestamp": now.isoformat(),
            "amount": amount,
            "payment_method": payment_method,
            "is_international": is_international,
            "merchant_category": merchant_category,
            "ip_address_risk_score": ip_risk,
            "device_trust_score": device_trust,
            "txn_count_last_24h": txn_count_24h,
            "location_change_flag": 1 if ip_risk > 0.7 else 0,
            "otp_success_rate_customer": 0.9,
            "past_fraud_count_customer": 0,
            "past_disputes_customer": 0,
            "merchant_historical_fraud_rate": merchant_hist_fraud,
            "hour_of_day": now.hour,
            "is_weekend": 1 if now.weekday() >= 5 else 0,
            "ip_address_country_match": 1,
            "customer_tenure_days": 365
        }])

        # [FIX] Unpack all three values from the new API Proxy method
        label, action, prob = scorer.predict_label_and_action(df_input)

        st.markdown("---")

        # 2. Dynamic Alert Banners
        if action == "HARD_BLOCK":
            st.error(f"🚨 **HIGH RISK DETECTED: {action}** | Fraud Probability: {prob:.2%}", icon="🚨")
        elif action in ["MANUAL_REVIEW", "OTP_VERIFICATION"]:
            st.warning(f"⚠️ **ELEVATED RISK: {action}** | Fraud Probability: {prob:.2%}", icon="⚠️")
        else:
            st.success(f"✅ **TRANSACTION APPROVED: {action}** | Fraud Probability: {prob:.2%}", icon="✅")

        # 3. Visual Outcome Layout
        colA, colB = st.columns([1, 1.5])

        with colA:
            st.markdown("<br>", unsafe_allow_html=True)
            
            action_color = "#ef4444" if label == 1 else "#10b981"
            html_content = (
                f"<b>Model Output:</b> {prob:.2%} Probability<br><br>"
                f"<b>Strict Threshold:</b> {threshold:.3f}<br><br>"
                f"<b>System Action:</b> <span style='color: {action_color}; font-weight: bold;'>{action}</span>"
            )
            
            info_card(
                "Decision Parameters",
                html_content,
                accent="warning" if label == 1 else "primary"
            )

        with colB:
            # Styled Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fraud Risk Score", 'font': {'size': 24}},
                delta={'reference': threshold * 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#ef4444" if label == 1 else "#10b981"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, threshold * 100], 'color': "rgba(16, 185, 129, 0.2)"},
                        {'range': [threshold * 100, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)

        # 4. Explainable AI (Data-Driven Marginal Contribution via Proxy)
        st.subheader("Model Decision Drivers (Explainability)")
        st.caption("Visualizing the exact mathematical impact of key features on the final risk score using marginal feature substitution via API.")
        
        safe_baselines = {
            "ip_address_risk_score": 0.0,
            "device_trust_score": 1.0,
            "amount": 500.0,  
            "txn_count_last_24h": 1,
            "merchant_historical_fraud_rate": 0.01
        }
        
        df_safe = df_input.copy()
        for feature, safe_val in safe_baselines.items():
            df_safe[feature] = safe_val
        
        # Use predict_proba via the proxy
        base_prob = scorer.predict_proba(df_safe)
        
        impacts = {}
        for feature, safe_val in safe_baselines.items():
            df_temp = df_safe.copy()
            df_temp[feature] = df_input[feature].iloc[0] 
            
            prob_with_feature = scorer.predict_proba(df_temp)
            impact = prob_with_feature - base_prob
            impacts[feature] = impact
            
        sorted_impacts = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)
        top_drivers = sorted_impacts[:3] 
        
        explained_sum = sum([imp for _, imp in top_drivers])
        other_impact = prob - (base_prob + explained_sum)
        
        x_values = [base_prob] + [imp for _, imp in top_drivers] + [other_impact, prob]
        y_labels = ["Safe Baseline"] + [f.replace("_", " ").title() for f, _ in top_drivers] + ["Other Interactions", "Final Score"]
        measures = ["absolute"] + ["relative"] * len(top_drivers) + ["relative", "total"]
        
        fig_waterfall = go.Figure(go.Waterfall(
            name="Risk Drivers", orientation="h",
            measure=measures,
            y=y_labels,
            x=x_values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#10b981"}}, 
            increasing={"marker": {"color": "#ef4444"}}, 
            totals={"marker": {"color": "#4C8BF5"}}
        ))
        
        fig_waterfall.update_layout(
            height=350, 
            margin=dict(l=10, r=20, t=30, b=10),
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed")
        )
        fig_waterfall.update_xaxes(tickformat=',.1%')
        
        st.plotly_chart(fig_waterfall, use_container_width=True)