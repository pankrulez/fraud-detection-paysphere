from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def render_live_scoring(scorer, threshold: float):

    st.title("Real-Time Risk Scoring Console")

    st.markdown(
        "Simulate incoming transactions and observe model probability, "
        "risk classification, and operational decision."
    )

    st.markdown("---")

    with st.form("txn_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            amount = st.number_input("Amount (₹)", 1.0, value=1200.0)
            payment_method = st.selectbox("Payment Method", ["UPI", "CARD", "NETBANKING", "WALLET"])
            merchant_category = st.selectbox("Merchant Category",
                                             ["Electronics", "Travel", "Fashion", "Gaming", "Grocery", "Utilities"])

        with col2:
            ip_risk = st.slider("IP Risk Score", 0.0, 1.0, 0.25)
            device_trust = st.slider("Device Trust Score", 0.0, 1.0, 0.8)
            txn_count_24h = st.number_input("Txn Count (24h)", 0, value=2)

        with col3:
            merchant_hist_fraud = st.slider("Merchant Fraud Rate", 0.0, 1.0, 0.05)
            is_international = st.selectbox("International?", [0, 1])

        submitted = st.form_submit_button("Score Transaction")

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
            "avg_amount_last_24h": 0,
            "merchant_diversity_last_7d": 0,
            "device_change_flag": 0,
            "location_change_flag": 0,
            "authentication_method": "OTP",
            "otp_success_rate_customer": 1,
            "past_fraud_count_customer": 0,
            "past_disputes_customer": 0,
            "merchant_historical_fraud_rate": merchant_hist_fraud,
            "hour_of_day": now.hour,
            "day_of_week": now.weekday(),
            "is_weekend": 1 if now.weekday() >= 5 else 0,
            "is_fraud": 0,
        }])

        prob = scorer.predict_proba(df_input)
        label, action = scorer.predict_label_and_action(df_input)

        st.markdown("---")
        st.subheader("Risk Assessment Output")

        colA, colB = st.columns([1, 2])

        with colA:
            st.metric("Fraud Probability", f"{prob:.2%}")
            st.metric("Decision Threshold", f"{threshold:.3f}")
            st.metric("Model Label", "FRAUD" if label == 1 else "GENUINE")
            st.metric("Recommended Action", action)

        with colB:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                gauge={"axis": {"range": [0, 100]}}
            ))
            st.plotly_chart(fig, use_container_width=True)