import streamlit as st


def render_overview(load_sample_data_fn):

    df = load_sample_data_fn()

    total_txn = len(df)
    fraud_count = int(df["is_fraud"].sum())
    fraud_rate = df["is_fraud"].mean() * 100

    st.markdown("""
    <div class="glass-card">
        <h2>Executive Overview</h2>
        <p>
        PaySphere is a real-time fraud detection system that transforms
        machine learning probabilities into actionable business decisions.
        The engine balances fraud prevention with customer experience
        through threshold-based decisioning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Operational Snapshot")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Transactions Analysed", f"{total_txn:,}")
    k2.metric("Fraud Cases", f"{fraud_count:,}")
    k3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    k4.metric("Model Type", "Tree-Based Ensemble")

    st.markdown("### System Capabilities")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="glass-card">
            <h4>Real-Time Risk Scoring</h4>
            <p>Probability-based fraud prediction with configurable decision threshold.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="glass-card">
            <h4>Explainable Risk Signals</h4>
            <p>Behavioral, device, network and temporal intelligence combined.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="glass-card">
            <h4>MLOps Ready Architecture</h4>
            <p>Versioned artifacts, reproducible preprocessing, CI-tested workflow.</p>
        </div>
        """, unsafe_allow_html=True)