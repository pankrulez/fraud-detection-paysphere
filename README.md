# 🛡️ PaySphere Fraud Risk Intelligence

![Pytest Status](https://github.com/pankrulez/fraud-detection-paysphere/actions/workflows/main.yml/badge.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Enterprise_UI-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3f4f75?logo=plotly&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end, real-time machine learning pipeline and enterprise-grade dashboard for detecting fraudulent digital payments (UPI, Cards, NetBanking, Wallets) under extreme class imbalance.

---

## 🚀 The Business Problem & Solution

Fraud in digital payments often represents less than 0.5% of total transaction volume, making it a "needle in a haystack" problem. Traditional rule-based engines generate too many false positives, causing high customer friction. 

**PaySphere Risk Intelligence** solves this by leveraging a tree-based ML classifier (RandomForest) equipped with SMOTE for imbalance handling. It goes beyond simple classification by providing a **production-ready Streamlit interface** featuring real-time explainable AI (XAI) and dynamic financial impact simulations.

### ✨ Key Upgrades & Features
- **Real-Time Scoring & Explainability:** A live interception console that not only predicts fraud probabilities but visually explains *why* via **Marginal Feature Substitution Waterfall charts**.
- **Financial Cost-Benefit Simulator:** Dynamic threshold tuning that calculates exact Rupee (₹) impacts in real-time (Fraud Prevented vs. Genuine Blocked).
- **Advanced Model Diagnostics:** Precision-Recall curves, ROC-AUC, and global feature correlation drivers to prove model robustness.
- **Actionability:** Automated CSV exports of flagged transactions for human-in-the-loop review.
- **Dynamic Artifact Registry:** Live inspection of `.joblib` model artifacts to ensure governance and provenance.

---

## 🖥️ Dashboard Walkthrough

![App Dashboard](reports/dashboard.png)

1. **Overview:** High-level KPIs, log-scaled financial outlier analysis, and fraud distribution across payment rails.
2. **Live Scoring:** Simulate real-time transactions with dynamic alert banners and pseudo-SHAP waterfall charts explaining the model's exact mathematical drivers.
3. **Analytics:** The Data Science control room. View confusion matrices, PR/ROC curves, global feature importances, and the Business Value Simulator. 
4. **Pipeline:** A technical walkthrough of the data ingestion, feature engineering, and class imbalance strategies, topped with a live Model Artifact Registry.

---

## 🧱 Project Architecture

```text
fraud-detection-paysphere/
├── app/
│   ├── app.py                  # Main Streamlit entrypoint (router)
│   ├── overview_view.py        # Executive KPIs & Outlier Analysis
│   ├── live_view.py            # Live scoring with Explainable AI (Waterfall)
│   ├── analytics_view.py       # ML Curves, Feature Drivers & Fin-Simulator
│   ├── pipeline_view.py        # Architecture details & Model Registry
│   └── ui_components.py        # Native Streamlit card components & styling
├── config/
│   ├── config.yaml             # Data paths, model, threshold config
│   └── logging.yaml            # Logging configuration
├── data/
│   ├── raw/                    # Input dataset (synthetic)
│   ├── interim/                # Cleaned data
│   └── processed/              # Features + predictions (test set)
├── models/
│   ├── artifacts/
│   │   └── fraud_model.joblib  # Trained model
│   └── encoders/
│       └── preprocessing.joblib# Encoders / scaler
├── src/
│   ├── data_ingestion/         # Load + validate + clean data
│   ├── features/               # Feature engineering + SMOTE
│   ├── modeling/               # Training pipeline & Inference Scorer
│   ├── pipeline/               # CLI entry to run full pipeline
│   ├── utils/                  # IO, schema validation, config utils
│   ├── logger.py               # Structured logging
│   └── exceptions.py           # Custom exception types
├── tests/                      # Pytest unit tests
├── requirements.txt
└── README.md
```

## 📦 Run It Locally
1️⃣ **Clone & Install Dependencies**

```bash
git clone [https://github.com/](https://github.com/)<your-username>/fraud-detection-paysphere.git

cd fraud-detection-paysphere

python -m venv .venv

source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

2️⃣ **Execute the ML Pipeline (Train the Model)**

```bash
python -m src.pipeline.run_pipeline
```
This will ingest data, engineer features, apply SMOTE, train the classifier, and save the production artifacts to `/models/`.

3️⃣ **Launch the Risk Intelligence Dashboard**

```bash
streamlit run app/app.py
```

## 🧠 Data Science Methodology

- **Feature Engineering**: Captured transaction velocity (1h, 24h, 7d windows), amount deviations from baselines, device trust scores, and historical merchant fraud rates.

- **Imbalance Strategy**: Utilized SMOTE (Synthetic Minority Over-sampling Technique) combined with rigorous evaluation prioritizing Precision-Recall AUC over standard accuracy.

- **Model Agnostic XAI**: Built custom drop-one marginal contribution logic to calculate feature importance in real-time without heavy dependency bloat.

---

## 🧪 Testing
The project maintains high reliability through automated unit testing covering data schemas, feature scaling, model training, and inference.

```bash
pytest
```

---

## 📌 Future Improvements

- Deploy API layer via FastAPI to decouple the model from the UI.

- Containerization using Docker for standardized deployment.

- Implement strictly continuous drift monitoring (e.g., EvidentlyAI).

- Migrate from pseudo-SHAP to formal shap TreeExplainer integration for deeper local explainability.

## 📄 License

MIT
