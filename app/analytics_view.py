# app/analytics_view.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def render_analytics(load_sample_data_fn, show_raw: bool, threshold: float, scorer):

    df = load_sample_data_fn()
    
    # ---------------------------------------------------
    # KPI STRIP (Real Model + Threshold Reactive)
    # ---------------------------------------------------
    st.markdown("### 📈 KPI Snapshot (Model-Driven)")

    # Run real model scoring
    proba = scorer.predict_proba(df)

    # If model returns 2D array, take fraud class probability
    if len(proba.shape) == 2:
        df["model_probability"] = proba[:, 1]
    else:
        df["model_probability"] = proba


    # Debug safe guard
    if df["model_probability"].max() == 0:
        st.warning("Model returned zero probabilities for all rows.")
        return

    df["predicted_fraud"] = (df["model_probability"] >= threshold).astype(int)

    actual_rate = df["is_fraud"].mean()
    predicted_rate = df["predicted_fraud"].mean()

    flagged_txns = df["predicted_fraud"].sum()
    actual_fraud_count = df["is_fraud"].sum()

    captured_fraud = df[
        (df["predicted_fraud"] == 1) & (df["is_fraud"] == 1)
    ].shape[0]

    false_positives = df[
        (df["predicted_fraud"] == 1) & (df["is_fraud"] == 0)
    ].shape[0]

    precision = (
        captured_fraud / flagged_txns
        if flagged_txns > 0 else 0
    )

    recall = (
        captured_fraud / actual_fraud_count
        if actual_fraud_count > 0 else 0
    )
    suggested_threshold = df["model_probability"].quantile(0.95)

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Actual Fraud Rate", f"{actual_rate:.2%}")
    k2.metric("Predicted Fraud Rate", f"{predicted_rate:.2%}")
    k3.metric("Precision", f"{precision:.2%}")
    k4.metric("Recall", f"{recall:.2%}")
    st.caption(f"95th percentile probability: {suggested_threshold:.3f}")

    st.markdown(
        f"""
        <div style="font-size:16px; color:#cbd5e1;">
        Current Decision Threshold: <b>{threshold:.2f}</b><br><br>
        • Precision reflects false-positive control (customer protection).<br>
        • Recall reflects fraud capture strength (loss prevention).<br>
        Adjusting the sidebar threshold directly changes this trade-off.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    
    # ---------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------
    st.markdown("### 🧮 Confusion Matrix")

    tp = df[(df["predicted_fraud"] == 1) & (df["is_fraud"] == 1)].shape[0]
    tn = df[(df["predicted_fraud"] == 0) & (df["is_fraud"] == 0)].shape[0]
    fp = df[(df["predicted_fraud"] == 1) & (df["is_fraud"] == 0)].shape[0]
    fn = df[(df["predicted_fraud"] == 0) & (df["is_fraud"] == 1)].shape[0]

    import numpy as np
    import plotly.figure_factory as ff

    matrix = np.array([[tn, fp],
                       [fn, tp]])
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        fig = ff.create_annotated_heatmap(
            z=matrix,
            x=["Predicted Genuine", "Predicted Fraud"],
            y=["Actual Genuine", "Actual Fraud"],
            colorscale="Blues",
            showscale=True,
        )

        fig.update_layout(
            height=420,
            title="Model Decision Breakdown",
            title_font_size=20,
        )

        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(
            """
            <div style="font-size:17px; color:#cbd5e1;">
            This matrix shows how many transactions fall into each category:
            <ul>
                <li><b style="color:#22c55e;">True Negatives (TN):</b> Genuine transactions correctly identified.</li>
                <li><b style="color:#ef4444;">False Positives (FP):</b> Genuine transactions incorrectly flagged as fraud.</li>
                <li><b style="color:#ef4444;">False Negatives (FN):</b> Fraudulent transactions missed by the model.</li>
                <li><b style="color:#22c55e;">True Positives (TP):</b> Fraudulent transactions correctly flagged.</li>
            </ul>
            The balance of these categories reflects the model's performance and the chosen threshold.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ---------------------------------------------------
    # FRAUD HEATMAP MATRIX
    # ---------------------------------------------------
    st.markdown("### 🔥 Fraud Heatmap Matrix (Hour × Payment Method)")

    pivot = (
        df.groupby(["hour_of_day", "payment_method"])["is_fraud"]
        .mean()
        .reset_index()
    )
    col1, col2 = st.columns([3, 2])
    
    with col1:

        heatmap = px.density_heatmap(
            pivot,
            x="hour_of_day",
            y="payment_method",
            z="is_fraud",
            color_continuous_scale="RdYlGn_r",
            height=420,
            title="Fraud Rate by Hour and Payment Method",
        )

        heatmap.update_layout(
            title_font_size=20,
            xaxis_title="Hour of Day",
            yaxis_title="Payment Method",
        )

        st.plotly_chart(heatmap, use_container_width=True)

    with col2:
        st.markdown(
            """
            <div style="font-size:17px; color:#cbd5e1;">
            This heatmap reveals high-risk combinations of transaction time and payment method.
            <ul>
                <li>Red zones indicate higher fraud rates, guiding real-time risk rules.</li>
                <li>For example, if UPI transactions spike in fraud during late hours, the system can apply stricter controls during those times.</li>
                <li>This matrix helps translate model insights into operational strategies.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    
    st.markdown(
        "<h2 style='font-size:28px;'>📊 Fraud Analytics & Behavioural Patterns</h2>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p style='font-size:15px; color:#cbd5e1;'>
        These charts demonstrate how the fraud model learns patterns from class imbalance,
        transaction amount behavior, temporal spikes, velocity signals, and device/network risk indicators.
        Each visualization reflects a feature signal that directly contributes to real-time scoring decisions.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if show_raw:
        st.markdown("#### Sample of cleaned transactions")
        st.dataframe(df.head(50))

    # -----------------------------
    # 1. Class Distribution
    # -----------------------------
    col1, col2 = st.columns([3, 2])

    with col1:
        fig = px.histogram(
            df,
            x="is_fraud",
            color="is_fraud",
            color_discrete_sequence=["#22c55e", "#ef4444"],
            title="Class Imbalance",
            height=380,
        )
        fig.update_layout(
            title_font_size=20,
            xaxis_title="Fraud Label (0 = Genuine, 1 = Fraud)",
            yaxis_title="Transaction Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        <div style="font-size:17px; line-height:1.6;">
        <b style="color:#ef4444;">Key Insight:</b><br><br>
        Fraud is extremely rare compared to genuine transactions.
        This severe imbalance is why techniques like 
        <span style="color:#ef4444; font-weight:600;">SMOTE</span> are applied.
        <br><br>
        The model must detect rare fraud events without overwhelming the system
        with false positives.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # -----------------------------
    # 2. Amount Distribution
    # -----------------------------
    col1, col2 = st.columns([3, 2])

    with col1:
        fig = px.box(
            df,
            x="is_fraud",
            y="amount",
            color="is_fraud",
            color_discrete_sequence=["#22c55e", "#ef4444"],
            title="Transaction Amount by Label",
            height=400,
        )
        fig.update_layout(title_font_size=20)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        <div style="font-size:17px; line-height:1.6;">
        <b style="color:#f59e0b;">Behavioral Pattern:</b><br><br>
        Fraud clusters around specific amount bands.
        <br><br>
        Common strategies:
        <ul>
            <li>Small test transactions before large attacks</li>
            <li>High-value rapid withdrawals</li>
        </ul>
        Amount is one of the strongest predictive features in fraud modeling.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # -----------------------------
    # 3. Fraud Rate by Payment Method
    # -----------------------------
    tmp_pm = df.groupby("payment_method")["is_fraud"].mean().reset_index()

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = px.bar(
            tmp_pm,
            x="payment_method",
            y="is_fraud",
            color="is_fraud",
            color_continuous_scale="RdYlGn_r",
            title="Fraud Rate by Payment Method",
            height=380,
        )
        fig.update_layout(title_font_size=20)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        <div style="font-size:17px; line-height:1.6;">
        <b style="color:#3b82f6;">Operational Insight:</b><br><br>
        Different payment rails have different fraud profiles.
        <br><br>
        Cards may have higher chargeback exposure, 
        while instant rails (UPI / wallets) may see velocity abuse.
        <br><br>
        The model captures these structural differences.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # -----------------------------
    # 4. Temporal Risk (Hour of Day)
    # -----------------------------
    tmp_hour = df.groupby("hour_of_day")["is_fraud"].mean().reset_index()

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = px.line(
            tmp_hour,
            x="hour_of_day",
            y="is_fraud",
            markers=True,
            title="Fraud Rate by Hour of Day",
            height=380,
        )
        fig.update_traces(line=dict(width=3))
        fig.update_layout(title_font_size=20)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        <div style="font-size:17px; line-height:1.6;">
        <b style="color:#ef4444;">Temporal Spike:</b><br><br>
        Fraud often increases during late-night or low-monitoring hours.
        <br><br>
        Organized fraud campaigns exploit reduced oversight windows.
        <br><br>
        Time-based features improve model sensitivity to these spikes.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # -----------------------------
    # 5. IP Risk vs Device Trust
    # -----------------------------
    sample = df.sample(min(2000, len(df)), random_state=42)

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = px.scatter(
            sample,
            x="ip_address_risk_score",
            y="device_trust_score",
            color="is_fraud",
            opacity=0.6,
            color_discrete_sequence=["#22c55e", "#ef4444"],
            title="IP Risk vs Device Trust",
            height=420,
        )
        fig.update_layout(title_font_size=20)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        <div style="font-size:17px; line-height:1.6;">
        <b style="color:#9333ea;">Network & Device Intelligence:</b><br><br>
        High IP risk + Low device trust is a strong fraud signal.
        <br><br>
        This pattern is typical of:
        <ul>
            <li>Bots & emulators</li>
            <li>Proxy or VPN abuse</li>
            <li>Account takeover attempts</li>
        </ul>
        Combining network and device signals significantly boosts detection power.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.info(
        "Charts on the left show quantitative patterns. "
        "Panels on the right explain how each signal contributes to the ML decision layer."
    )