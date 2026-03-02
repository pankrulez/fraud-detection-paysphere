import os
import joblib
import streamlit as st
from app.ui_components import info_card

def render_pipeline():
    st.title("Fraud Detection Pipeline Architecture")
    st.caption("Technical specification of the end-to-end fraud detection system.")

    st.write(
        """
        The fraud detection pipeline transforms raw digital payment transactions into 
        calibrated fraud risk decisions under extreme class imbalance conditions 
        (<0.5% fraud rate). The system balances fraud detection performance 
        with customer experience and operational constraints.
        """
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Define the pipeline steps cleanly
    steps = [
        {
            "step": 1, "title": "Data Ingestion & Validation", "accent": "primary",
            "obj": "Ensure structured, consistent, and analysis-ready transaction data before modeling.",
            "meth": ["Loaded structured transaction datasets", "Validated schema consistency across fields", "Handled missing values", "Parsed timestamp features for time-based analysis"],
            "out": "Clean, validated dataset ready for feature engineering."
        },
        {
            "step": 2, "title": "Behavioral Feature Engineering", "accent": "success",
            "obj": "Extract high-signal fraud indicators using behavioral and contextual signals.",
            "meth": ["Engineered transaction velocity features (24-hour window)", "Computed customer spending behavior metrics", "Calculated merchant diversity measures", "Extracted temporal indicators"],
            "out": "Feature matrix capturing transaction-level and behavioral risk signals."
        },
        {
            "step": 3, "title": "Class Imbalance Strategy", "accent": "warning",
            "obj": "Address extreme fraud rarity to prevent majority-class bias in modeling.",
            "meth": ["Analyzed fraud distribution and skewness", "Applied resampling techniques where required", "Optimized evaluation metrics toward precision-recall trade-off"],
            "out": "Balanced training setup reducing bias toward non-fraud predictions."
        },
        {
            "step": 4, "title": "Model Training & Optimization", "accent": "info",
            "obj": "Train a classification model capable of discriminating fraudulent activity.",
            "meth": ["Implemented tree-based ensemble classifier", "Validated using confusion matrix metrics", "Evaluated precision-recall trade-offs", "Calibrated probabilities"],
            "out": "Calibrated fraud probability model suitable for threshold-based decisioning."
        },
        {
            "step": 5, "title": "Real-Time Risk Scoring", "accent": "neutral",
            "obj": "Convert model probability outputs into operational fraud decisions.",
            "meth": ["Generated transaction-level fraud probabilities", "Mapped probabilities to decision thresholds", "Defined Allow / Review / Block logic", "Integrated scoring into live UI"],
            "out": "Operational fraud scoring engine enabling business-aligned decision control."
        },
        {
            "step": 6, "title": "Monitoring & Governance", "accent": "governance",
            "obj": "Ensure system reliability, maintainability, and reproducibility.",
            "meth": ["Versioned model artifacts", "Maintained modular project architecture", "Integrated automated testing", "Prepared deployment-ready structure"],
            "out": "Governance-ready fraud detection system with reproducible model lifecycle."
        }
    ]

    # Dynamically render using the unified UI component
    # Dynamically render using the unified UI component
    for s in steps:
        meth_html = "".join([f"<li style='margin-bottom: 4px;'>{m}</li>" for m in s['meth']])
        
        # Formatted as a single continuous string (no explicit line breaks) 
        # to prevent Streamlit's Markdown parser from breaking the div tags
        content_html = (
            f"<p style='margin-bottom: 8px;'><b>Objective:</b> {s['obj']}</p>"
            f"<p style='margin-bottom: 4px;'><b>Methodology:</b></p>"
            f"<ul style='margin-top: 0; padding-left: 20px;'>{meth_html}</ul>"
            f"<p style='margin-top: 8px;'><b>Output:</b> <span style='color: #10b981; font-weight: 500;'>{s['out']}</span></p>"
        )
        
        info_card(f"Step {s['step']}: {s['title']}", content_html, accent=s['accent'])
    
    st.markdown("---")
    st.subheader("📦 Production Model Registry")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    model_path = os.path.join(project_root, "models", "artifacts", "fraud_model.joblib")
    
    if os.path.exists(model_path):
        # Dynamically inspect the real model artifact
        file_size_kb = os.path.getsize(model_path) / 1024
        
        try:
            model = joblib.load(model_path)
            model_type = type(model).__name__
            
            # Try to get feature count (works for most sklearn/xgboost models)
            if hasattr(model, "n_features_in_"):
                n_features = model.n_features_in_
            elif hasattr(model, "feature_names_in_"):
                n_features = len(model.feature_names_in_)
            else:
                n_features = "Dynamic"
                
            registry_html = f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px;">
                <div>
                    <p style="margin:0; color:#a1a1aa; font-size:0.9rem;">Artifact File</p>
                    <p style="margin:0; font-weight:600; color:#e5e7eb;">fraud_model.joblib</p>
                </div>
                <div>
                    <p style="margin:0; color:#a1a1aa; font-size:0.9rem;">Algorithm Family</p>
                    <p style="margin:0; font-weight:600; color:#4C8BF5;">{model_type}</p>
                </div>
                <div>
                    <p style="margin:0; color:#a1a1aa; font-size:0.9rem;">Input Features</p>
                    <p style="margin:0; font-weight:600; color:#10b981;">{n_features}</p>
                </div>
                <div>
                    <p style="margin:0; color:#a1a1aa; font-size:0.9rem;">File Size</p>
                    <p style="margin:0; font-weight:600; color:#B08968;">{file_size_kb:.1f} KB</p>
                </div>
            </div>
            """
            info_card("Current Active Model Artifact", registry_html, accent="neutral")
            
        except Exception as e:
            st.warning(f"Model file exists but couldn't be loaded for inspection: {e}")
    else:
        st.error(f"Model artifact not found at {model_path}")