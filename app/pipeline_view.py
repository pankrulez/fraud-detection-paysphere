# app/pipeline_view.py
import streamlit as st


def render_pipeline():

    st.markdown("### 🧬 End-to-End Fraud Detection Pipeline")

    st.markdown(
        "This page is a slide-free architecture walkthrough of the PaySphere fraud engine, "
        "from raw payments to real-time actions."
    )

    steps = [
        (
            "1", "Data Ingestion & Validation", "#3b82f6",
            "Raw transactions from UPI, cards, net banking, and wallets are loaded from "
            "<code>transactions_fraud.csv</code>. We validate schema, enforce business rules (amount "
            "positivity, temporal ranges, binary flags), and persist clean data to "
            "<code>data/interim/transactions_clean.csv</code> for repeatable experiments.",
            "Is the data reliable enough to take risk decisions on?",
        ),
        (
            "2", "Feature Engineering & Behavioural Signals", "#9333ea",
            "We transform raw fields into richer signals: customer spend deviation, short- "
            "and medium-term velocity, device sharing, and historical fraud/dispute ratios. "
            "A combined risk index aggregates IP reputation, device trust, customer history, "
            "and merchant risk into a single score for the model.",
            "What does this transaction look like in the context of the customer, device, and merchant history?",
        ),
        (
            "3", "Imbalance Handling & Model Training", "#ef4444",
            "Because fraud is extremely rare, we apply SMOTE to balance the training set. "
            "A tree-based classifier (RandomForest / XGBoost) is trained on the engineered "
            "features. We track ROC AUC and PR AUC, but emphasise precision and recall at "
            "a chosen threshold to reflect fraud loss vs customer friction trade-offs.",
            "How well can the model separate fraud from genuine behaviour under real class imbalance?",
        ),
        (
            "4", "Model Serialization & Artifacts", "#3b82f6",
            "The trained model and preprocessing stack are versioned and saved using joblib "
            "<code>models/artifacts/fraud_model.joblib</code>, "
            "<code>models/encoders/preprocessing.joblib</code>."
            "The FraudScorer class encapsulates loading, preprocessing, scoring, and mapping "
            "probabilities to actions, making downstream integration simple.",
            "Can we reproduce this model later and reliably deploy the same logic?",
        ),
        (
            "5", "Real-Time Scoring & Decisioning", "#f59e0b",
            "For each incoming transaction, we apply the same feature logic, generate a fraud "
            "probability, and compare it to the configurable threshold (see sidebar). "
            "Probability bands are mapped to business actions (HARD_BLOCK, OTP_CHALLENGE, "
            "SOFT_REVIEW, ALLOW) that plug directly into the payment gateway or analyst queues.",
            "Given this risk score, what should actually happen to the transaction right now?",
        ),
        (
            "6", "Testing, CI/CD & Streamlit UI", "#22c55e",
            "pytest tests cover ingestion, feature engineering, training, and inference. "
            "The repo is CI-ready (GitHub Actions) so tests run on every push. "
            "The Streamlit app serves both as an internal tooling surface for analysts "
            "and as a lightweight production UI for early deployments and demos.",
            "How do we keep this fraud engine reliable as data, code, and models evolve?",
        ),
    ]

    # ---- Timeline Rendering ----
    for step, title, color, body, question in steps:

        container = st.container()

        with container:
            col1, col2 = st.columns([1, 12])

            # Step circle
            with col1:
                st.markdown(
                    f"""
                    <div style="
                        width:45px;
                        height:45px;
                        border-radius:50%;
                        background:{color};
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        color:white;
                        font-weight:bold;
                        box-shadow:0 4px 14px rgba(0,0,0,0.4);
                    ">
                        {step}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Card content
            with col2:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <h4 style="margin-top:0; color:{color};">{title}</h4>
                        <p style="color:#cbd5e1;">{body}</p>
                        <p style="color:#94a3b8; font-size:0.9rem;">
                            <b style = "color:{color};">Answers questions like:</b> "{question}"
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)

    st.info(
        "Walk through these steps live while switching to Live Scoring and Analytics "
        "to show how architecture, ML, and business actions connect."
    )