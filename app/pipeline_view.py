import os
import joblib
import streamlit as st
from app.ui_components import info_card

def render_pipeline():
    # 1. PREMIUM UI STYLING (Professional Stepper & Glassmorphism)
    st.markdown("""
        <style>
        /* Container for the overall page */
        .reportview-container {
            background: #0F172A;
        }
        /* Gradient Card Style */
        div[data-testid="stExpander"], div.stMarkdown div {
            border-radius: 12px;
        }
        .pipeline-card {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid #334155;
            margin-bottom: 20px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        .step-number {
            background: #3B82F6;
            color: white;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 12px;
        }
        .registry-container {
            background: #111827;
            border-radius: 12px;
            border-left: 4px solid #4F46E5;
            padding: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ Fraud Detection Pipeline")
    st.caption("Enterprise-grade ML architecture for high-frequency payment streams.")

    st.write(
        """
        The PaySphere pipeline transforms raw transaction logs into calibrated risk decisions. 
        It is architected to solve the **Extreme Class Imbalance** problem while maintaining 
        sub-50ms inference latency.
        """
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Content remains unchanged as requested
    steps = [
        {
            "step": 1, "title": "Data Ingestion & Validation", "accent": "#3B82F6",
            "obj": "Ensure structured, consistent, and analysis-ready transaction data before modeling.",
            "meth": ["Loaded structured transaction datasets", "Validated schema consistency across fields", "Handled missing values", "Parsed timestamp features"],
            "out": "Clean, validated dataset ready for feature engineering."
        },
        {
            "step": 2, "title": "Behavioral Feature Engineering", "accent": "#10B981",
            "obj": "Extract high-signal fraud indicators using behavioral and contextual signals.",
            "meth": ["Engineered transaction velocity (24h)", "Computed spending behavior metrics", "Calculated merchant diversity", "Extracted temporal indicators"],
            "out": "Feature matrix capturing behavioral risk signals."
        },
        {
            "step": 3, "title": "Class Imbalance Strategy", "accent": "#F59E0B",
            "obj": "Address extreme fraud rarity to prevent majority-class bias in modeling.",
            "meth": ["Analyzed fraud distribution skewness", "Applied SMOTE resampling where required", "Optimized toward Precision-Recall trade-off"],
            "out": "Balanced training setup reducing non-fraud bias."
        },
        {
            "step": 4, "title": "Model Training & Optimization", "accent": "#6366F1",
            "obj": "Train a classification model capable of discriminating fraudulent activity.",
            "meth": ["Implemented tree-based ensemble classifier", "Validated confusion matrix metrics", "Calibrated probability outputs"],
            "out": "Calibrated probability model for threshold-based decisioning."
        },
        {
            "step": 5, "title": "Real-Time Risk Scoring", "accent": "#EC4899",
            "obj": "Convert model probability outputs into operational fraud decisions.",
            "meth": ["Generated transaction-level probabilities", "Mapped to decision thresholds", "Defined Allow / Review / Block logic"],
            "out": "Operational scoring engine with dynamic decision control."
        },
        {
            "step": 6, "title": "Monitoring & Governance", "accent": "#94A3B8",
            "obj": "Ensure system reliability, maintainability, and reproducibility.",
            "meth": ["Versioned model artifacts", "Modular project architecture", "Integrated automated testing"],
            "out": "Governance-ready system with reproducible model lifecycle."
        }
    ]

    # Render Steps as Premium Stepper Cards
    for s in steps:
        with st.container():
            st.markdown(f"""
                <div class="pipeline-card">
                    <div style="display: flex; align-items: center; margin-bottom: 12px;">
                        <span class="step-number" style="background-color: {s['accent']};">{s['step']}</span>
                        <h3 style="margin: 0; color: white;">{s['title']}</h3>
                    </div>
                    <p style="color: #94A3B8; font-size: 0.95rem;"><b>Objective:</b> {s['obj']}</p>
                    <ul style="color: #CBD5E1; font-size: 0.9rem;">
                        {"".join([f"<li>{m}</li>" for m in s['meth']])}
                    </ul>
                    <div style="background: rgba(16, 185, 129, 0.1); padding: 8px 12px; border-radius: 6px; border-left: 3px solid #10B981;">
                        <span style="color: #10B981; font-weight: 600; font-size: 0.85rem;">OUTPUT:</span>
                        <span style="color: #E5E7EB; font-size: 0.85rem;"> {s['out']}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # 2. ENHANCED MODEL REGISTRY
    st.markdown("---")
    st.subheader("📦 Production Model Registry")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    model_path = os.path.join(project_root, "models", "artifacts", "fraud_model.joblib")
    
    if os.path.exists(model_path):
        file_size_kb = os.path.getsize(model_path) / 1024
        try:
            model = joblib.load(model_path)
            model_type = type(model).__name__
            n_features = model.n_features_in_ if hasattr(model, "n_features_in_") else "Dynamic"
                
            st.markdown(f"""
                <div class="registry-container">
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <p style="margin:0; color:#64748B; font-size:0.8rem; text-transform: uppercase;">Artifact</p>
                            <p style="margin:0; font-weight:600; color:#F8FAFC;">fraud_model.joblib</p>
                        </div>
                        <div>
                            <p style="margin:0; color:#64748B; font-size:0.8rem; text-transform: uppercase;">Algorithm</p>
                            <p style="margin:0; font-weight:600; color:#3B82F6;">{model_type}</p>
                        </div>
                        <div>
                            <p style="margin:0; color:#64748B; font-size:0.8rem; text-transform: uppercase;">Features</p>
                            <p style="margin:0; font-weight:600; color:#10B981;">{n_features}</p>
                        </div>
                        <div>
                            <p style="margin:0; color:#64748B; font-size:0.8rem; text-transform: uppercase;">Size</p>
                            <p style="margin:0; font-weight:600; color:#F59E0B;">{file_size_kb:.1f} KB</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Registry Error: {e}")
    else:
        st.error("Model artifact not detected.")