import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from app.ui_components import render_threshold_explanation

def render_analytics(load_sample_data_fn, show_raw: bool, threshold: float, scorer):
    # 1. STYLE INJECTION (Dull Card Gradients)
    st.markdown("""
        <style>
        [data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; color: #F8FAFC; }
        div[data-testid="column"] {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            padding: 20px; border-radius: 12px; border: 1px solid #334155;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # 2. DATA PREP (Using the API Proxy)
    if 'scored_df' not in st.session_state:
        df_raw = load_sample_data_fn()
        
        sample_to_score = df_raw.head(500).copy()
        
        # --- SCHEMA ALIGNMENT ---
        # Ensure the batch data perfectly matches the Pydantic API expectations
        
        # 1. Ensure required string/ID fields exist
        if "customer_id" not in sample_to_score.columns: sample_to_score["customer_id"] = "CUST_X"
        if "device_id" not in sample_to_score.columns: sample_to_score["device_id"] = "DEV_X"
        if "merchant_id" not in sample_to_score.columns: sample_to_score["merchant_id"] = "MERCH_X"
        if "timestamp" not in sample_to_score.columns: sample_to_score["timestamp"] = datetime.utcnow().isoformat()
        
        # 2. Ensure required numeric fields exist
        defaults = {
            "is_international": 0, "is_weekend": 0, "past_fraud_count_customer": 0,
            "past_disputes_customer": 0, "txn_count_last_24h": 0, "customer_tenure_days": 0,
            "location_change_flag": 0, "otp_success_rate_customer": 1.0, 
            "ip_address_country_match": 1, "hour_of_day": 12, "day_of_week": 3,
            "merchant_historical_fraud_rate": 0.05, "ip_address_risk_score": 0.1,
            "device_trust_score": 0.9, "amount": 100.0, "payment_method": "credit_card",
            "merchant_category": "retail"
        }
        for col, default_val in defaults.items():
            if col not in sample_to_score.columns:
                sample_to_score[col] = default_val

        # Ensure string types for categories to avoid float/int typing errors
        sample_to_score['payment_method'] = sample_to_score['payment_method'].astype(str)
        sample_to_score['merchant_category'] = sample_to_score['merchant_category'].astype(str)
        sample_to_score['customer_id'] = sample_to_score['customer_id'].astype(str)
        sample_to_score['device_id'] = sample_to_score['device_id'].astype(str)
        sample_to_score['merchant_id'] = sample_to_score['merchant_id'].astype(str)

        # ---- API BATCH SCORING ---- #

        with st.spinner("Scoring 500 transactions instantly via Batch API..."):
            probabilities = scorer.predict_proba_batch(sample_to_score)
            
            sample_to_score['model_probability'] = probabilities
            st.session_state.scored_df = sample_to_score

    df = st.session_state.scored_df
    df['predicted_fraud'] = (df['model_probability'] >= threshold).astype(int)

    # 3. GRADIENT CARDS: FINANCIAL SIMULATOR
    st.title("📊 Fraud Analytics & Business Impact")
    render_threshold_explanation(threshold)
    s1, s2, s3 = st.columns(3)
    
    saved = df[(df['predicted_fraud'] == 1) & (df['is_fraud'] == 1)]['amount'].sum()
    loss = df[(df['predicted_fraud'] == 0) & (df['is_fraud'] == 1)]['amount'].sum()
    friction = df[(df['predicted_fraud'] == 1) & (df['is_fraud'] == 0)]['amount'].sum()

    s1.metric(
        "Fraud Prevented", 
        f"₹{saved:,.0f}", 
        help="The total value of confirmed fraudulent transactions successfully blocked by the model. This represents direct capital loss prevention."
        )
    s2.metric(
        "Fraud Missed", 
        f"₹{loss:,.0f}", 
        delta_color="inverse",
        help="The value of fraudulent transactions that the model failed to flag (False Negatives). This is the direct cost of 'leaked' fraud."
        )
    s3.metric(
        "Friction Cost", 
        f"₹{friction:,.0f}", 
        delta_color="inverse",
        help="The revenue at risk from genuine customers whose transactions were flagged as suspicious (False Positives). High friction can lead to customer churn."
        )

    st.markdown("---")

    # Row 1: Category & Amount
    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        cat_data = df.groupby('merchant_category')['is_fraud'].mean().reset_index()
        fig1 = px.bar(cat_data, x='merchant_category', y='is_fraud', 
                      title="Fraud Rate by Category", color='is_fraud', 
                      color_continuous_scale='Viridis')
        st.plotly_chart(fig1, use_container_width=True)
    with r1_col2:
        fig2 = px.violin(df, x="is_fraud", y="amount", box=True, 
                         color="is_fraud", color_discrete_sequence=['#3B82F6', '#EF4444'],
                         title="Amount Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Method & Risk Correlation
    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        method_data = df.groupby('payment_method')['is_fraud'].mean().reset_index()
        fig3 = px.bar(method_data, x='payment_method', y='is_fraud', 
                      title="Fraud Rate by Method", color='payment_method',
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig3, use_container_width=True)
    with r2_col2:
        corrs = df.select_dtypes(include=[np.number]).corr()['is_fraud'].drop('is_fraud', errors='ignore').sort_values()
        fig4 = px.bar(x=corrs.values, y=corrs.index, orientation='h', 
                      title="Risk Signal Strength", color=corrs.values,
                      color_continuous_scale='RdBu_r')
        st.plotly_chart(fig4, use_container_width=True)

    # Row 3: Risk Clustering & Velocity
    r3_col1, r3_col2 = st.columns(2)
    with r3_col1:
        fig5 = px.scatter(df, x="ip_address_risk_score", y="device_trust_score", 
                          color="is_fraud", title="Risk Clustering",
                          color_discrete_sequence=['#3B82F6', '#EF4444'], opacity=0.6)
        st.plotly_chart(fig5, use_container_width=True)
    with r3_col2:
        fig6 = px.histogram(df, x="txn_count_last_24h", color="is_fraud", 
                            barmode="group", title="24h Velocity Impact",
                            color_discrete_sequence=['#3B82F6', '#EF4444'])
        st.plotly_chart(fig6, use_container_width=True)

    if show_raw:
        st.dataframe(df[df['predicted_fraud'] == 1], use_container_width=True)