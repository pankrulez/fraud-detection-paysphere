import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from app.ui_components import chart_card

def render_overview(load_sample_data_fn):

    df = load_sample_data_fn()

    # -------------------------
    # Core Dataset Metrics
    # -------------------------
    total_txn = len(df)
    fraud_count = int(df["is_fraud"].sum())
    fraud_rate = df["is_fraud"].mean()
    genuine_count = total_txn - fraud_count

    avg_amount = df["amount"].mean()
    median_amount = df["amount"].median()
    max_amount = df["amount"].max()

    fraud_exposure = df[df["is_fraud"] == 1]["amount"].sum()

    st.title("PaySphere Fraud Risk Intelligence Dashboard")

    st.markdown("""
    **Company Context** PaySphere Digital Payments Pvt. Ltd. operates at national scale across UPI, cards,
    net banking and wallets. Fraud rates are low in percentage terms but high in financial impact.
    This dashboard translates fraud analytics into operational intelligence.
    """)

    st.markdown("---")

    # =========================
    # KPI STRIP
    # =========================
    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Transactions Analysed (Sample)", f"{total_txn:,}")
    k2.metric("Confirmed Fraud Cases", f"{fraud_count:,}")
    k3.metric("Fraud Rate", f"{fraud_rate:.3%}")
    k4.metric("Genuine Transactions", f"{genuine_count:,}")

    st.markdown("---")

    k5, k6, k7, k8 = st.columns(4)

    k5.metric("Avg Transaction Value", f"₹{avg_amount:,.0f}")
    k6.metric("Median Transaction Value", f"₹{median_amount:,.0f}")
    k7.metric("Max Transaction Observed", f"₹{max_amount:,.0f}")
    k8.metric("Fraud Financial Exposure (Sample)", f"₹{fraud_exposure:,.0f}")

    st.markdown("---")

    # =========================
    # DISTRIBUTION + FRAUD PROFILE
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Distribution (Imbalance)")

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(
            y=["Transactions"], x=[genuine_count], 
            name="Genuine", orientation='h', marker=dict(color="#10b981")
        ))
        fig_dist.add_trace(go.Bar(
            y=["Transactions"], x=[fraud_count], 
            name="Fraud", orientation='h', marker=dict(color="#ef4444")
        ))

        fig_dist.update_layout(
            barmode='stack', height=250, margin=dict(t=30, b=0, l=0, r=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})
        st.caption("Fraud represents a very small fraction of total volume — highlighting severe class imbalance (<0.5% typical in production).")

    with col2:
        st.subheader("Fraud Rate by Payment Rail")

        fraud_by_pm = df.groupby("payment_method")["is_fraud"].mean().reset_index().sort_values(by="is_fraud", ascending=False)
        fig2 = px.bar(fraud_by_pm, x="payment_method", y="is_fraud", color="is_fraud", color_continuous_scale="Reds")
        
        fig2.update_layout(
            height=250, margin=dict(t=30, b=0, l=0, r=0),
            xaxis_title="", yaxis_title="Fraud Rate", coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        fig2.update_yaxes(tickformat=',.1%')
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.caption("Different payment rails exhibit structurally different risk patterns.")

    st.markdown("---")

    # =========================
    # FINANCIAL OUTLIER ANALYSIS (NEW)
    # =========================
    st.subheader("Financial Outlier Analysis")
    
    plot_df = df.copy()
    plot_df["Label"] = plot_df["is_fraud"].map({0: "Genuine", 1: "Fraud"})
    
    fig_box = px.box(
        plot_df, x="Label", y="amount", color="Label",
        color_discrete_map={"Genuine": "#10b981", "Fraud": "#ef4444"},
        points="outliers"
    )
    
    fig_box.update_layout(
        yaxis_type="log", # CRITICAL: Log scale reveals the extremes without squashing the middle
        yaxis_title="Transaction Amount (₹) - Log Scale", xaxis_title="",
        margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    
    chart_card(
        "Transaction Value Distributions (Log-Scaled)", 
        "Notice how fraudulent transactions cluster around different extremes compared to genuine behavior.", 
        fig_box, accent="info", height=350
    )

    st.markdown("---")

    # =========================
    # HIGH RISK SEGMENTS
    # =========================
    st.subheader("Risk Concentration Analysis")

    col3, col4 = st.columns(2)

    with col3:
        high_risk_hours = df.groupby("hour_of_day")["is_fraud"].mean().reset_index().sort_values(by="is_fraud", ascending=False).head(5)
        st.markdown("**Top 5 High-Risk Hours**")
        styled_hours = high_risk_hours.style.background_gradient(cmap="Reds", subset=["is_fraud"]).format({"is_fraud": "{:.2%}"})
        st.dataframe(styled_hours, use_container_width=True, hide_index=True)

    with col4:
        high_risk_merchants = df.groupby("merchant_category")["is_fraud"].mean().reset_index().sort_values(by="is_fraud", ascending=False).head(5)
        st.markdown("**Top 5 High-Risk Merchant Categories**")
        styled_merchants = high_risk_merchants.style.background_gradient(cmap="Reds", subset=["is_fraud"]).format({"is_fraud": "{:.2%}"})
        st.dataframe(styled_merchants, use_container_width=True, hide_index=True)

    st.markdown("---")

    # =========================
    # STRATEGIC INTERPRETATION
    # =========================
    st.info(f"""
    **Strategic Risk Interpretation:**
    * Fraud rate in the analysed dataset is **{fraud_rate:.3%}**, reflecting extreme imbalance.  
    * Financial exposure from confirmed fraud in this sample totals **₹{fraud_exposure:,.0f}**.  
    * Risk is unevenly distributed across payment rails and time windows.  
    * Detection strategy must balance reducing false negatives (chargebacks) with minimising false positives (customer friction).
    * **Conclusion:** This necessitates probability-based scoring with threshold tuning, not rule-based flagging alone.
    """)