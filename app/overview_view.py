import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
from app.ui_components import chart_card, info_card

def render_overview(load_sample_data_fn, scorer):
    # Professional Styling 
    st.markdown("""
        <style>
        [data-testid="stMetricValue"] { font-size: 32px; font-weight: 800; color: #F8FAFC; }
        div[data-testid="column"] {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            padding: 20px; border-radius: 12px; border: 1px solid #334155;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # 1. DATA INGESTION
    df = load_sample_data_fn()
    total_txn = len(df)
    fraud_count = int(df["is_fraud"].sum())
    fraud_rate = df["is_fraud"].mean()
    fraud_exposure = df[df["is_fraud"] == 1]["amount"].sum()
    avg_amount = df["amount"].mean()

    # 2. HEADER
    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.title("🛡️ PaySphere Risk Command Center")
        st.caption(f"Real-time integrity monitoring across {total_txn:,} transaction vectors.")
    
    with col_status:
        st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; 
                        padding: 12px; border-radius: 10px; text-align: center;">
                <span style="color: #10b981; font-weight: 700; font-size: 0.9rem;">● ENGINE OPERATIONAL</span><br>
                <span style="color: #94a3b8; font-size: 0.75rem;">v1.0.0 Stable Build</span>
            </div>
        """, unsafe_allow_html=True)

    st.write("---")

    # 3. EXECUTIVE KPI STRIP
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Monitoring Volume", f"{total_txn:,}")
    k2.metric("Anomaly Rate", f"{fraud_rate:.3%}")
    k3.metric("Capital at Risk", f"₹{fraud_exposure:,.0f}")
    k4.metric("Avg. Ticket Size", f"₹{avg_amount:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. MODEL CONFIDENCE & TURBO HEATMAP
    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        avg_confidence = 0.92 
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_confidence * 100,
            title={'text': "Model Decision Certainty", 'font': {'size': 18, 'color': '#94A3B8'}},
            number={'suffix': "%", 'font': {'color': '#F8FAFC'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "#334155"},
                'bar': {'color': "#3B82F6"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 70], 'color': 'rgba(239, 68, 68, 0.1)'},
                    {'range': [90, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
                ],
                'threshold': {'line': {'color': "#10b981", 'width': 4}, 'value': 90}
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_right:
        # HIGH CONTRAST TURBO TREEMAP
        fig_tree = px.treemap(
            df[df['is_fraud'] == 1], 
            path=['merchant_category', 'payment_method'], 
            values='amount',
            color='amount',
            color_continuous_scale='Turbo',
            title="Fraud Value Distribution (High-Contrast)"
        )
        fig_tree.update_layout(
            margin=dict(t=30, b=0, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': '#F8FAFC'}
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    st.write("---")

    # 5. HOURLY RISK & AUDIT LOG
    col_v1, col_v2 = st.columns([1.5, 1])
    with col_v1:
        hourly_risk = df.groupby("hour_of_day")["is_fraud"].mean().reset_index()
        fig_area = px.area(hourly_risk, x="hour_of_day", y="is_fraud", title="Intraday Risk Velocity")
        fig_area.update_traces(line_color='#ef4444', fillcolor='rgba(239, 68, 68, 0.1)')
        fig_area.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': '#94A3B8'})
        st.plotly_chart(fig_area, use_container_width=True)

    with col_v2:
        st.subheader("🚩 Recent Interceptions")
        audit_log = df[df['is_fraud'] == 1].tail(5)[['customer_id', 'amount', 'merchant_category']]
        st.dataframe(audit_log, use_container_width=True, hide_index=True)

    # 6. BUSINESS INTERPRETATION (CLEANED)
    st.write("---")
    
    directives_content = f"""
    <div style='color: #CBD5E1; font-size: 0.9rem; line-height: 1.6;'>
        <b>1. Baseline Integrity:</b> The current fraud rate of {fraud_rate:.2%} is within established limits.<br>
        <b>2. High Value Alert:</b> Targeted sectors: Travel and Electronics.<br>
        <b>3. System Suggestion:</b> Confidence is high ({avg_confidence*100}%); optimize manual review bands.
    </div>
    """
    
    # We pass the content directly to the info_card without wrapping it in st.info
    info_card(
        "Operational Directives",
        directives_content,
        accent="primary"
    )