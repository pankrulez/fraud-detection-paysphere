import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from app.ui_components import chart_card, render_threshold_explanation

def render_analytics(load_sample_data_fn, show_raw: bool, threshold: float, scorer):
    # Header
    col_t, col_s = st.columns([3, 1])
    with col_t:
        st.title("📊 Business Impact Analytics")
        st.caption("ROI Simulation and behavioral risk distribution across 25,000 samples.")
    
    render_threshold_explanation(threshold)
    st.write("---")

    # 1. DATA PREP (Using session state to avoid re-scoring)
    if 'scored_df' not in st.session_state:
        with st.spinner("Scoring batch for ROI analysis..."):
            df_raw = load_sample_data_fn()
            sample = df_raw.head(25000).copy()
            # In a real app, you'd call scorer.predict_proba_batch(sample) here
            # For now, we assume probabilities exist or simulate them for the UI demo
            sample['prob'] = scorer.predict_proba_batch(sample)
            st.session_state.scored_df = sample

    df = st.session_state.scored_df
    df['pred'] = (df['prob'] >= threshold).astype(int)

    # 2. FINANCIAL IMPACT STRIP
    saved = df[(df['pred'] == 1) & (df['is_fraud'] == 1)]['amount'].sum()
    leakage = df[(df['pred'] == 0) & (df['is_fraud'] == 1)]['amount'].sum()
    friction = df[(df['pred'] == 1) & (df['is_fraud'] == 0)]['amount'].sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Fraud Prevented", f"₹{saved:,.0f}", help="Total capital loss averted.")
    m2.metric("Revenue Leakage", f"₹{leakage:,.0f}", delta="False Negatives", delta_color="inverse")
    m3.metric("Friction Cost", f"₹{friction:,.0f}", delta="False Positives", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. TURBO HEATMAPS
    col_left, col_right = st.columns(2)

    with col_left:
        # High Contrast Turbo Treemap
        fig_tree = px.treemap(
            df[df['prob'] > threshold], 
            path=['merchant_category', 'payment_method'], 
            values='amount', color='prob',
            color_continuous_scale='Turbo',
            title="Concentration of Blocked Capital"
        )
        chart_card("Risk Concentration", "Where the model is focusing its blocking strategy.", fig_tree)

    with col_right:
        # Feature Correlation with Risk
        numeric_df = df.select_dtypes(include=[np.number])
        corrs = numeric_df.corr()['prob'].drop(['prob', 'pred', 'is_fraud'], errors='ignore').sort_values()
        fig_corr = px.bar(x=corrs.values, y=corrs.index, orientation='h', color=corrs.values, color_continuous_scale='Turbo')
        chart_card("Signal Strength", "Which features are driving the high risk scores.", fig_corr)

    if show_raw:
        st.subheader("🚩 High-Risk Manifest")
        st.dataframe(df[df['pred'] == 1].sort_values('prob', ascending=False), use_container_width=True)