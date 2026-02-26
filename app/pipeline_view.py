import streamlit as st


def render_pipeline():

    st.title("Fraud Detection Architecture & Governance")

    st.markdown("""
    This pipeline is designed for large-scale digital payments where fraud detection,
    customer experience, and regulatory compliance intersect.
    """)

    st.markdown("---")

    stages = {
        "1. Data Layer":
            "Transaction ingestion, validation, schema enforcement, data cleaning.",
        "2. Feature Engineering":
            "Behavioral velocity, device risk, IP intelligence, merchant history.",
        "3. Imbalance Strategy":
            "SMOTE and threshold calibration for severe class imbalance.",
        "4. Model Training":
            "Tree-based ensemble classifier optimised for precision-recall trade-off.",
        "5. Decision Layer":
            "Probability-to-action mapping (Hard Block / OTP / Review / Allow).",
        "6. Monitoring & CI":
            "Versioned artifacts, pytest validation, reproducible deployments."
    }

    for title, desc in stages.items():
        st.markdown(f"### {title}")
        st.write(desc)
        st.markdown("---")