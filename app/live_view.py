from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def render_live_scoring(scorer, threshold: float):

    st.markdown("### Live Transaction Risk Console")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    with st.form("txn_form"):

        c1, c2, c3 = st.columns(3)

        with c1:
            amount = st.number_input("Amount (₹)", 1.0, value=1200.0)
            payment_method = st.selectbox("Payment Method", ["UPI", "CARD", "NETBANKING", "WALLET"])
            merchant_category = st.selectbox("Merchant Category",
                                             ["Electronics", "Travel", "Fashion", "Gaming", "Grocery", "Utilities"])
            is_international = st.selectbox("Is International?", [0, 1])

        with c2:
            ip_risk = st.slider("IP Risk Score", 0.0, 1.0, 0.25)
            device_trust = st.slider("Device Trust Score", 0.0, 1.0, 0.8)
            device_change_flag = st.selectbox("Device Change", [0, 1])
            location_change_flag = st.selectbox("Location Change", [0, 1])

        with c3:
            txn_count_24h = st.number_input("Txn Count (24h)", 0, value=2)
            avg_amount_24h = st.number_input("Avg Amount (24h)", 0.0, value=900.0)
            otp_success_rate = st.slider("OTP Success Rate", 0.0, 1.0, 0.95)
            past_fraud_count = st.number_input("Past Fraud Count", 0, value=0)
            past_disputes = st.number_input("Past Disputes", 0, value=0)
            merchant_hist_fraud = st.slider("Merchant Fraud Rate", 0.0, 1.0, 0.05)

        submitted = st.form_submit_button("Score Transaction")

    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        now = datetime.now()

        df_input = pd.DataFrame([{
            "transaction_id": 0,
            "customer_id": 0,
            "device_id": 0,
            "merchant_id": 0,
            "timestamp": now.isoformat(),
            "amount": amount,
            "payment_method": payment_method,
            "is_international": is_international,
            "merchant_category": merchant_category,
            "ip_address_risk_score": ip_risk,
            "device_trust_score": device_trust,
            "txn_count_last_24h": txn_count_24h,
            "avg_amount_last_24h": avg_amount_24h,
            "merchant_diversity_last_7d": 2,
            "device_change_flag": device_change_flag,
            "location_change_flag": location_change_flag,
            "authentication_method": "OTP",
            "otp_success_rate_customer": otp_success_rate,
            "past_fraud_count_customer": past_fraud_count,
            "past_disputes_customer": past_disputes,
            "merchant_historical_fraud_rate": merchant_hist_fraud,
            "hour_of_day": now.hour,
            "day_of_week": now.weekday(),
            "is_weekend": 1 if now.weekday() >= 5 else 0,
            "is_fraud": 0,
        }])

        prob = scorer.predict_proba(df_input)
        label, action = scorer.predict_label_and_action(df_input)

        risk_band = (
            "LOW RISK" if prob < 0.03 else
            "MEDIUM RISK" if prob < threshold else
            "HIGH RISK"
        )

        band_color = "#22c55e" if risk_band == "LOW RISK" else \
                     "#f59e0b" if risk_band == "MEDIUM RISK" else "#ef4444"

        st.markdown(f"""
        <div class="glass-card">
            <h2>Risk Assessment</h2>
            <h1 style="color:{band_color};">{risk_band}</h1>
            <p>Fraud Probability: <b>{prob:.2%}</b></p>
            <p>Decision Threshold: <b>{threshold:.3f}</b></p>
            <p>Recommended Action: <b>{action}</b></p>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": band_color},
            },
        ))

        st.plotly_chart(fig, use_container_width=True)