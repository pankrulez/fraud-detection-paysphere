import streamlit as st


def render_pipeline():

    st.markdown("### End-to-End Fraud Detection Architecture")

    steps = [
        ("1", "Data Ingestion & Validation"),
        ("2", "Feature Engineering"),
        ("3", "Imbalance Handling & Training"),
        ("4", "Model Serialization"),
        ("5", "Real-Time Decisioning"),
        ("6", "Testing & CI/CD"),
    ]

    cols = st.columns(2)

    for i, (step, title) in enumerate(steps):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="glass-card">
                <h3>Step {step}: {title}</h3>
                <p>
                Modular and reproducible pipeline stage with versioned artifacts
                and production-ready deployment logic.
                </p>
            </div>
            """, unsafe_allow_html=True)