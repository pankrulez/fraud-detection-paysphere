# PaySphere Online Payment Fraud Detection

![Pytest Status](https://github.com/pankrulez/fraud-detection-paysphere/actions/workflows/main.yml/badge.svg)
![Python 3.10+](https://img.shields.io)
![Streamlit](https://img.shields.io)
![Scikit-Learn](https://img.shields.io)
![Pandas](https://img.shields.io)
![Code style: black](https://img.shields.io)
![License: MIT](https://img.shields.io)


Real-time fraud detection system for digital payments (UPI, cards, wallets).

End-to-end ML pipeline → trained model → production-style artifacts → Streamlit app for live scoring.

---

## 🚀 What This Project Demonstrates

- End-to-end ML pipeline (data → features → model → artifacts)

- Class imbalance handling (SMOTE)

- Threshold-based fraud decisioning

- Clean modular architecture (src/ structure)

- Artifact versioning (model + preprocessing)

- Interactive fraud analytics dashboard

- Unit tests with pytest

---

## 🧱 Project Structure

```text
fraud-detection-paysphere/
├── app/
│   ├── app.py                  # Main Streamlit entrypoint (router)
│   ├── overview_view.py        # Overview tab
│   ├── live_view.py            # Live scoring tab
│   ├── analytics_view.py       # Analytics & plots tab
│   └── pipeline_view.py        # Project pipeline tab
├── config/
│   ├── config.yaml             # Data paths, model, threshold config
│   └── logging.yaml            # Logging configuration
├── data/
│   ├── raw/
│   │   └── transactions_fraud.csv        # Input dataset (synthetic)
│   ├── interim/
│   │   └── transactions_clean.csv        # Cleaned data
│   └── processed/
│       └── transactions_features.csv     # Features + predictions (test set)
├── models/
│   ├── artifacts/
│   │   └── fraud_model.joblib            # Trained model
│   └── encoders/
│       └── preprocessing.joblib          # Encoders / scaler
├── src/
│   ├── data_ingestion/
│   │   └── ingestion.py                  # Load + validate + clean data
│   ├── features/
│   │   └── feature_engineering.py        # Feature engineering + SMOTE
│   ├── modeling/
│   │   ├── train.py                      # train_pipeline()
│   │   └── inference.py                  # FraudScorer
│   ├── pipeline/
│   │   └── run_pipeline.py               # CLI entry to run full pipeline
│   ├── utils/
│   │   ├── io_utils.py                   # read/write CSV, save models
│   │   ├── validation_utils.py           # schema + business rule checks
│   │   └── config_utils.py               # load YAML config
│   ├── logger.py                         # structured logging
│   └── exceptions.py                     # custom exception types
├── tests/
│   ├── test_data_ingestion.py
│   ├── test_feature_engineering.py
│   ├── test_model_training.py
│   └── test_inference.py
├── requirements.txt
└── README.md
```

## 📦 Run It Locally
1️⃣ Install

```bash
git clone https://github.com/<your-username>/fraud-detection-paysphere.git

cd fraud-detection-paysphere

python -m venv .venv

source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

2️⃣ Train Model

```bash
python -m src.pipeline.run_pipeline
```

Artifacts saved:

- `models/artifacts/fraud_model.joblib`

- `models/encoders/preprocessing.joblib`

3️⃣ Launch App

```bash
streamlit run app/app.py
```

## 🖥 Streamlit App Features

- **Live Scoring**

    - Enter transaction attributes

    - Get fraud probability

    - Business action: ALLOW / REVIEW / BLOCK

- **Fraud Analytics**

    - Class distribution

    - Temporal fraud patterns

    - Risk signal visualization

    - Behavioural insights

- **Pipeline Walkthrough**

    - Structured explanation of each ML stage

---

## 🧠 Feature Engineering Highlights

- Transaction velocity signals

- Amount deviation from customer baseline

- Device & IP risk indicators

- Historical fraud patterns

- Combined risk index

---

## 📊 Model

- Tree-based classifier (RandomForest)

- SMOTE for imbalance

- ROC AUC & PR AUC evaluation

- Configurable fraud decision threshold

---

## 🧪 Running Tests

```bash
pytest
```
Covers ingestion, feature engineering, training, and inference.

---

## 📌 Future Improvements

- Model explainability (SHAP)

- Drift monitoring

- API layer (FastAPI)

- Multiple model comparison

## 📄 License

MIT
