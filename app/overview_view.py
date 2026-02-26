import streamlit as st
import plotly.express as px
import pandas as pd


def render_overview(load_sample_data_fn):

    df = load_sample_data_fn()

    # -------------------------
    # Core Dataset Metrics
    # -------------------------
    total_txn = len(df)
    fraud_count = int(df["is_fraud"].sum())
    fraud_rate = df["is_fraud"].mean() * 100
    genuine_count = total_txn - fraud_count

    avg_amount = df["amount"].mean()
    median_amount = df["amount"].median()
    max_amount = df["amount"].max()

    fraud_exposure = df[df["is_fraud"] == 1]["amount"].sum()

    st.title("PaySphere Fraud Risk Intelligence Dashboard")

    st.markdown("""
    **Company Context**  
    PaySphere Digital Payments Pvt. Ltd. operates at national scale across UPI, cards,
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
    k3.metric("Fraud Rate", f"{fraud_rate:.3f}%")
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
        st.subheader("Transaction Distribution")

        dist_df = df["is_fraud"].value_counts().reset_index()
        dist_df.columns = ["Label", "Count"]
        dist_df["Label"] = dist_df["Label"].map({0: "Genuine", 1: "Fraud"})

        fig = px.pie(
            dist_df,
            names="Label",
            values="Count",
            hole=0.55,
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Fraud represents a very small fraction of total volume — "
            "highlighting severe class imbalance (<0.5% typical in production)."
        )

    with col2:
        st.subheader("Fraud Rate by Payment Rail")

        fraud_by_pm = (
            df.groupby("payment_method")["is_fraud"]
            .mean()
            .reset_index()
            .sort_values(by="is_fraud", ascending=False)
        )

        fig2 = px.bar(
            fraud_by_pm,
            x="payment_method",
            y="is_fraud",
        )
        fig2.update_layout(
            height=360,
            xaxis_title="Payment Method",
            yaxis_title="Fraud Rate",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            "Different payment rails exhibit structurally different risk patterns."
        )

    st.markdown("---")

    # =========================
    # HIGH RISK SEGMENTS
    # =========================
    st.subheader("Risk Concentration Analysis")

    col3, col4 = st.columns(2)

    with col3:
        high_risk_hours = (
            df.groupby("hour_of_day")["is_fraud"]
            .mean()
            .reset_index()
            .sort_values(by="is_fraud", ascending=False)
            .head(5)
        )

        st.markdown("**Top 5 High-Risk Hours**")
        st.dataframe(high_risk_hours, use_container_width=True)

    with col4:
        high_risk_merchants = (
            df.groupby("merchant_category")["is_fraud"]
            .mean()
            .reset_index()
            .sort_values(by="is_fraud", ascending=False)
            .head(5)
        )

        st.markdown("**Top 5 High-Risk Merchant Categories**")
        st.dataframe(high_risk_merchants, use_container_width=True)

    st.markdown("---")

    # =========================
    # STRATEGIC INTERPRETATION
    # =========================
    st.subheader("Strategic Risk Interpretation")

    st.markdown(f"""
    • Fraud rate in the analysed dataset is **{fraud_rate:.3f}%**, reflecting extreme imbalance.  
    • Financial exposure from confirmed fraud in this sample totals **₹{fraud_exposure:,.0f}**.  
    • Risk is unevenly distributed across payment rails and time windows.  
    • Detection strategy must balance:
        - Reducing false negatives (chargebacks & loss)
        - Minimising false positives (customer friction & revenue loss)
    • This necessitates probability-based scoring with threshold tuning, 
      not rule-based flagging alone.
    """)