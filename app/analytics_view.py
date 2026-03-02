import os
import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.figure_factory as ff
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from app.ui_components import chart_card

def render_analytics(load_sample_data_fn, show_raw: bool, threshold: float, scorer):

    df = load_sample_data_fn()

    st.title("Model Performance & Risk Analytics")
    st.markdown("Comprehensive evaluation of classification quality, financial impact, and behavioral signals.")
    st.markdown("---")

    # =============================
    # MODEL SCORING
    # =============================
    proba = scorer.predict_proba(df)

    if len(proba.shape) == 2:
        df["model_probability"] = proba[:, 1]
    else:
        df["model_probability"] = proba

    df["predicted_fraud"] = (df["model_probability"] >= threshold).astype(int)

    # Core Metrics
    y_true = df["is_fraud"]
    y_scores = df["model_probability"]
    y_pred = df["predicted_fraud"]

    tp = df[(y_pred == 1) & (y_true == 1)].shape[0]
    tn = df[(y_pred == 0) & (y_true == 0)].shape[0]
    fp = df[(y_pred == 1) & (y_true == 0)].shape[0]
    fn = df[(y_pred == 0) & (y_true == 1)].shape[0]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # =============================
    # FINANCIAL IMPACT SIMULATOR
    # =============================
    st.markdown("### 💰 Financial Impact Simulator")
    st.caption("Real-time calculation of business value based on the selected decision threshold in the sidebar.")
    
    # Calculate exact financial impact using the 'amount' column
    saved_amount = df[(y_pred == 1) & (y_true == 1)]["amount"].sum()
    loss_amount = df[(y_pred == 0) & (y_true == 1)]["amount"].sum()
    friction_amount = df[(y_pred == 1) & (y_true == 0)]["amount"].sum()
    
    f1, f2, f3 = st.columns(3)
    
    with f1:
        with st.container(border=True):
            st.metric("🛡️ Fraud Prevented (Saved)", f"₹{saved_amount:,.0f}", help="True Positives * Transaction Amount")
    with f2:
        with st.container(border=True):
            st.metric("💸 Fraud Missed (Loss)", f"₹{loss_amount:,.0f}", delta_color="inverse", help="False Negatives * Transaction Amount")
    with f3:
        with st.container(border=True):
            st.metric("⚠️ Genuine Blocked (Friction)", f"₹{friction_amount:,.0f}", delta_color="inverse", help="False Positives * Transaction Amount")
        
    st.markdown("<br>", unsafe_allow_html=True)

    # =============================
    # PERFORMANCE SUMMARY
    # =============================
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Precision", f"{precision:.2%}")
    k2.metric("Recall", f"{recall:.2%}")
    k3.metric("True Positives", tp)
    k4.metric("False Positives", fp)
    st.markdown("<br>", unsafe_allow_html=True)

    # =============================
    # ROW 1: OUTCOMES & DISTRIBUTION
    # =============================
    col_a, col_b = st.columns(2)

    with col_a:
        matrix = np.array([[tn, fp], [fn, tp]])
        fig_cm = ff.create_annotated_heatmap(
            z=matrix, x=["Pred Genuine", "Pred Fraud"], y=["Actual Genuine", "Actual Fraud"],
            colorscale="Blues", showscale=False
        )
        chart_card("Confusion Matrix", f"Evaluated at threshold: {threshold:.3f}", fig_cm, accent="primary", height=320)

    with col_b:
        fig1 = px.histogram(df, x="model_probability", color="is_fraud", nbins=50, barmode="overlay")
        fig1.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
        chart_card("Probability Distribution", "Separation between genuine and fraud scores.", fig1, accent="info", height=320)

    # =============================
    # ROW 2: DATA SCIENCE CURVES & GLOBAL IMPORTANCE
    # =============================
    col_c, col_d = st.columns(2)

    with col_c:
        # Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall_vals, y=precision_vals, mode='lines', name=f'PR (AUC = {pr_auc:.3f})', line=dict(color='#10b981', width=3)))
        fig_pr.update_layout(xaxis_title='Recall', yaxis_title='Precision', margin=dict(t=10, b=10, l=10, r=10))
        
        chart_card("Precision-Recall Curve", f"Average Precision: {pr_auc:.3f}. Crucial metric for imbalanced data.", fig_pr, accent="success", height=320)

    with col_d:
        # Dynamically build the path to the processed data (from app/ to data/processed/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        processed_data_path = os.path.join(project_root, "data", "processed", "transactions_features.csv")
        
        try:
            # Load the fully engineered numeric dataset
            features_df = pd.read_csv(processed_data_path)
            
            # Calculate correlation against the actual 'is_fraud' label
            corr = features_df.corrwith(features_df["is_fraud"]).drop("is_fraud", errors="ignore").fillna(0)
            
            # Get the top 8 most highly correlated features
            top_corr = corr.abs().sort_values(ascending=False).head(8).index
            plot_data = corr[top_corr].sort_values(ascending=True)
            
            fig_imp = go.Figure(go.Bar(
                x=plot_data.values,
                y=[str(x).replace("_", " ").title() for x in plot_data.index],
                orientation='h',
                marker=dict(
                    color=plot_data.values,
                    colorscale="RdBu_r",
                    cmin=-0.6, cmax=0.6, # Locks 0 exactly in the center (white)
                    colorbar=dict(title="Correlation")
                )
            ))
            
            fig_imp.update_layout(
                xaxis_title="Correlation with Fraud Label", 
                yaxis_title="", 
                margin=dict(t=10, b=10, l=10, r=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            
            chart_card("Global Feature Drivers", "Top engineered features driving fraud.", fig_imp, accent="warning", height=320)
            
        except FileNotFoundError:
            st.error(f"Could not find {processed_data_path}. Please check the file path.")      
        
    # =============================
    # ROW 3: BEHAVIORAL INSIGHTS
    # =============================
    col_e, col_f = st.columns(2)

    with col_e:
        tmp_hour = df.groupby("hour_of_day")["is_fraud"].mean().reset_index()
        fig4 = px.line(tmp_hour, x="hour_of_day", y="is_fraud", markers=True)
        fig4.update_yaxes(tickformat=',.1%')
        chart_card("Risk by Hour", "Average fraud rate across the 24-hour cycle.", fig4, accent="neutral", height=320)

    with col_f:
        pivot = df.groupby(["hour_of_day", "payment_method"])["is_fraud"].mean().reset_index()
        fig5 = px.density_heatmap(pivot, x="hour_of_day", y="payment_method", z="is_fraud", color_continuous_scale="Reds")
        chart_card("Fraud Heatmap", "Risk clusters by hour and payment method.", fig5, accent="neutral", height=320)

    # =============================
    # ACTIONABILITY & RAW DATA
    # =============================
    if show_raw:
        st.markdown("---")
        st.markdown("### 📋 Operational Action: Review Flagged Transactions")
        st.caption("Export the transactions isolated by the model for manual review by the fraud team.")
        
        # Filter for transactions the model flagged as fraud based on the current threshold
        flagged_df = df[df["predicted_fraud"] == 1].copy()
        
        if not flagged_df.empty:
            st.warning(f"⚠️ **{len(flagged_df):,}** transactions flagged as fraud at the current threshold ({threshold:.3f}).")
            
            # Convert dataframe to CSV for the download button
            csv = flagged_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Flagged Transactions (CSV)",
                data=csv,
                file_name=f"flagged_fraud_txns_threshold_{threshold:.3f}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
        else:
            st.success("✅ No transactions flagged as fraud at the current threshold.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display the styled raw sample
        st.markdown("### Sample Scored Data")
        styled_df = df.head(50).style\
            .background_gradient(cmap="Reds", subset=["model_probability", "ip_address_risk_score"])\
            .highlight_max(subset=["is_fraud"], color="#7f1d1d")\
            .format({"model_probability": "{:.2%}", "amount": "₹{:,.2f}", "ip_address_risk_score": "{:.2f}"})
        st.dataframe(styled_df, use_container_width=True)