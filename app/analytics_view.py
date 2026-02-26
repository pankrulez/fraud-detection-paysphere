import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import plotly.figure_factory as ff


# =============================
# STANDARD CHART HEIGHT
# =============================
CHART_HEIGHT = 420


def section_title(title):
    st.markdown(f"## {title}")
    st.markdown("---")


def render_analytics(load_sample_data_fn, show_raw: bool, threshold: float, scorer):

    df = load_sample_data_fn()

    st.title("Model Performance & Risk Analytics")
    st.markdown("Comprehensive evaluation of classification quality and behavioural risk signals.")
    st.markdown(" ")

    # =============================
    # MODEL SCORING
    # =============================
    proba = scorer.predict_proba(df)

    if len(proba.shape) == 2:
        df["model_probability"] = proba[:, 1]
    else:
        df["model_probability"] = proba

    df["predicted_fraud"] = (df["model_probability"] >= threshold).astype(int)

    tp = df[(df["predicted_fraud"] == 1) & (df["is_fraud"] == 1)].shape[0]
    tn = df[(df["predicted_fraud"] == 0) & (df["is_fraud"] == 0)].shape[0]
    fp = df[(df["predicted_fraud"] == 1) & (df["is_fraud"] == 0)].shape[0]
    fn = df[(df["predicted_fraud"] == 0) & (df["is_fraud"] == 1)].shape[0]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # =============================
    # PERFORMANCE SUMMARY
    # =============================
    section_title("Performance Summary")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Precision", f"{precision:.2%}")
    k2.metric("Recall", f"{recall:.2%}")
    k3.metric("True Positives", tp)
    k4.metric("False Positives", fp)

    # =============================
    # CONFUSION MATRIX
    # =============================
    section_title("Confusion Matrix")

    matrix = np.array([[tn, fp],
                       [fn, tp]])

    fig_cm = ff.create_annotated_heatmap(
        z=matrix,
        x=["Pred Genuine", "Pred Fraud"],
        y=["Actual Genuine", "Actual Fraud"],
        colorscale="Blues",
        showscale=False,
    )

    fig_cm.update_layout(height=CHART_HEIGHT)
    st.plotly_chart(fig_cm, use_container_width=True)

    # =============================
    # CLASS DISTRIBUTION
    # =============================
    section_title("Class Distribution")

    fig1 = px.histogram(
        df,
        x="is_fraud",
        color="is_fraud",
        height=CHART_HEIGHT,
    )

    st.plotly_chart(fig1, use_container_width=True)

    # =============================
    # AMOUNT DISTRIBUTION
    # =============================
    section_title("Transaction Amount by Fraud Label")

    fig2 = px.box(
        df,
        x="is_fraud",
        y="amount",
        color="is_fraud",
        height=CHART_HEIGHT,
    )

    st.plotly_chart(fig2, use_container_width=True)

    # =============================
    # FRAUD RATE BY PAYMENT METHOD
    # =============================
    section_title("Fraud Rate by Payment Method")

    tmp_pm = df.groupby("payment_method")["is_fraud"].mean().reset_index()

    fig3 = px.bar(
        tmp_pm,
        x="payment_method",
        y="is_fraud",
        height=CHART_HEIGHT,
    )

    st.plotly_chart(fig3, use_container_width=True)

    # =============================
    # FRAUD RATE BY HOUR
    # =============================
    section_title("Fraud Rate by Hour of Day")

    tmp_hour = df.groupby("hour_of_day")["is_fraud"].mean().reset_index()

    fig4 = px.line(
        tmp_hour,
        x="hour_of_day",
        y="is_fraud",
        markers=True,
        height=CHART_HEIGHT,
    )

    st.plotly_chart(fig4, use_container_width=True)

    # =============================
    # RISK HEATMAP
    # =============================
    section_title("Fraud Heatmap (Hour × Payment Method)")

    pivot = (
        df.groupby(["hour_of_day", "payment_method"])["is_fraud"]
        .mean()
        .reset_index()
    )

    fig5 = px.density_heatmap(
        pivot,
        x="hour_of_day",
        y="payment_method",
        z="is_fraud",
        height=CHART_HEIGHT,
    )

    st.plotly_chart(fig5, use_container_width=True)

    # =============================
    # NETWORK & DEVICE RISK
    # =============================
    section_title("IP Risk vs Device Trust")

    sample = df.sample(min(2000, len(df)), random_state=42)

    fig6 = px.scatter(
        sample,
        x="ip_address_risk_score",
        y="device_trust_score",
        color="is_fraud",
        opacity=0.6,
        height=CHART_HEIGHT,
    )

    st.plotly_chart(fig6, use_container_width=True)

    if show_raw:
        section_title("Sample Scored Data")
        st.dataframe(df.head(50), use_container_width=True)