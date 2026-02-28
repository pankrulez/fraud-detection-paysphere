import streamlit as st


# =====================================================
# GLOBAL CARD STYLING (REAL TARGET)
# =====================================================
st.markdown("""
<style>

/* Reduce overall page vertical padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Compact card styling */
div[data-testid="stVerticalBlock"] > div {
    background-color: #151b26;
    border-radius: 12px;
    padding: 20px !important;
    margin-bottom: 24px !important;
}

/* Compact headings */
h2 {
    margin-top: 0.2rem !important;
    margin-bottom: 0.6rem !important;
}

h3 {
    margin-top: 0.8rem !important;
    margin-bottom: 0.3rem !important;
}

/* Reduce paragraph spacing */
p {
    margin-bottom: 0.5rem !important;
}

/* Reduce bullet spacing */
ul {
    margin-top: 0.2rem !important;
    margin-bottom: 0.6rem !important;
}

li {
    margin-bottom: 0.2rem !important;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# STAGE SECTION
# =====================================================
def stage_section(step, title, objective, methodology, output):
    step_colors = {
        1: "#4C8BF5", 2: "#5CA27F", 3: "#B08968",
        4: "#7E8CE0", 5: "#5DA9A6", 6: "#9A7AA0",
    }
    accent = step_colors.get(step, "#4C8BF5")
    methodology_html = "".join([f"<li>{m}</li>" for m in methodology])

    html = f"""
<div style="
    background-color:#151b26;
    padding:24px;
    border-radius:12px;
    border-left:4px solid {accent};
    border-top:1px solid rgba(255,255,255,0.06);
    border-right:1px solid rgba(255,255,255,0.06);
    border-bottom:1px solid rgba(255,255,255,0.06);
    margin-bottom:28px;
">
    <h2 style="color:{accent}; margin-bottom:14px; font-weight:600;">
        Step {step}: {title}
    </h2>

    <p><strong>Objective:</strong> {objective}</p>

    <h3 style="margin-top:18px;">Methodology</h3>
    <ul>{methodology_html}</ul>

    <p style="margin-top:18px;"><strong>Output:</strong> {output}</p>

</div>
"""
    st.html(html)


# =====================================================
# MAIN RENDER
# =====================================================
def render_pipeline():

    st.title("Fraud Detection Pipeline Architecture")
    st.caption("Technical specification of the end-to-end fraud detection system.")

    # st.divider()

    # st.markdown("## System Overview")
    st.write(
        """
        The fraud detection pipeline transforms raw digital payment transactions into 
        calibrated fraud risk decisions under extreme class imbalance conditions 
        (<0.5% fraud rate). The system balances fraud detection performance 
        with customer experience and operational constraints.
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)

    stage_section(
        1,
        "Data Ingestion & Validation",
        "Ensure structured, consistent, and analysis-ready transaction data before modeling.",
        [
            "Loaded structured transaction datasets",
            "Validated schema consistency across fields",
            "Handled missing values",
            "Parsed timestamp features for time-based analysis",
        ],
        "Clean, validated dataset ready for feature engineering.",
    )

    stage_section(
        2,
        "Behavioral Feature Engineering",
        "Extract high-signal fraud indicators using behavioral and contextual signals.",
        [
            "Engineered transaction velocity features (24-hour window)",
            "Computed customer spending behavior metrics",
            "Calculated merchant diversity measures",
            "Extracted temporal indicators (hour, weekend flags)",
        ],
        "Feature matrix capturing transaction-level and behavioral risk signals.",
    )

    stage_section(
        3,
        "Class Imbalance Strategy",
        "Address extreme fraud rarity to prevent majority-class bias in modeling.",
        [
            "Analyzed fraud distribution and skewness",
            "Applied resampling techniques where required",
            "Optimized evaluation metrics toward precision-recall trade-off",
            "Conducted threshold sensitivity analysis",
        ],
        "Balanced training setup reducing bias toward non-fraud predictions.",
    )

    stage_section(
        4,
        "Model Training & Optimization",
        "Train a classification model capable of discriminating fraudulent activity.",
        [
            "Implemented tree-based ensemble classifier",
            "Validated using confusion matrix metrics",
            "Evaluated precision-recall trade-offs",
            "Calibrated fraud probability outputs",
        ],
        "Calibrated fraud probability model suitable for threshold-based decisioning.",
    )

    stage_section(
        5,
        "Real-Time Risk Scoring",
        "Convert model probability outputs into operational fraud decisions.",
        [
            "Generated transaction-level fraud probabilities",
            "Mapped probabilities to decision thresholds",
            "Defined Allow / Review / Block logic",
            "Integrated scoring into live interface",
        ],
        "Operational fraud scoring engine enabling business-aligned decision control.",
    )

    stage_section(
        6,
        "Monitoring & Governance",
        "Ensure system reliability, maintainability, and reproducibility.",
        [
            "Versioned model artifacts",
            "Maintained modular project architecture",
            "Integrated automated testing (pytest)",
            "Prepared deployment-ready structure",
        ],
        "Governance-ready fraud detection system with reproducible model lifecycle.",
    )