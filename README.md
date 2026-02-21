# PaySphere Online Payment Fraud Detection

![Pytest Status](https://github.com/pankrulez/fraud-detection-paysphere/actions/workflows/<WORKFLOW_FILENAME>/badge.svg)


Real-time fraud detection system for a digital payments provider (UPI, cards, net banking, wallets).  
The project implements an end‑to‑end ML pipeline — from raw transactions to a Streamlit app for live scoring and fraud analytics.

---

## 🚀 Key Features

- **End‑to‑end ML pipeline**
  - Data ingestion and schema validation
  - Feature engineering with behavioural and risk‑based signals
  - Class‑imbalance handling using SMOTE
  - Model training, evaluation, and threshold tuning

- **Production‑style architecture**
  - Clear module structure under `src/`
  - Versioned model & preprocessing artifacts (`joblib`)
  - Inference wrapper (`FraudScorer`) for real‑time scoring

- **Interactive Streamlit app**
  - **Overview**: business and technical context
  - **Live Scoring**: score a single transaction and get recommended action
  - **Analytics & Plots**: class balance, temporal patterns, behavioural signals
  - **Project Pipeline**: step‑by‑step architecture walkthrough

- **Testing & CI‑ready**
  - `pytest` tests for ingestion, features, training, inference
  - Structure designed to plug into GitHub Actions for CI

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

## 📦 Installation
1. Clone the repository
```bash
git clone https://github.com/<your-username>/fraud-detection-paysphere.git

cd fraud-detection-paysphere
```

2. Create and activate a virtual environment
```bash
python -m venv .venv

# Windows

.venv\Scripts\activate

# macOS / Linux

source .venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## 🧪 Running Tests
The project uses `pytest` for unit/integration tests.

```bash
pytest
```

This will run:

- `test_data_ingestion.py` – schema checks & business rules on the input CSV

- `test_feature_engineering.py` – behavioural / risk feature creation

- `test_model_training.py` – end‑to‑end training pipeline with SMOTE

- `test_inference.py` – loading artifacts and scoring a single transaction

All tests should pass once you’ve run the training pipeline at least once (see below).

## 🔁 Training Pipeline (Data → Model → Artifacts)
To run the full ML pipeline (ingestion → features → training → artifacts):

```bash
python -m src.pipeline.run_pipeline
```
This will:

1. Read raw data from `data/raw/transactions_fraud.csv`.

2. Validate schema & rules and save a cleaned version to:

- `data/interim/transactions_clean.csv`

3. Engineer features (velocity, spend deviation, history, combined risk index) and handle class imbalance with SMOTE.

4. Train a tree‑based model (e.g., RandomForest) and evaluate:

- ROC AUC

- PR AUC

- Precision & recall at configured threshold

5. Save outputs:

- Test‑set features + predictions → `data/processed/transactions_features.csv`

- Model artifact → `models/artifacts/fraud_model.joblib`

- Preprocessing pipeline → `models/encoders/preprocessing.joblib`

These artifacts are then used by the Streamlit app for interactive scoring.

## 🖥️ Running the Streamlit App (Local)
From the project root:

```bash
streamlit run app/app.py
```

The app exposes four views (in the sidebar):

1. **Overview**

- Business motivation for fraud detection

- Sample dataset stats (transactions, fraud rate)

- High‑level business, ML, and MLOps highlights

2. **Live Scoring**

- Form for entering a single transaction’s attributes:

    - Amount, payment method, merchant category, international flag

    - IP address risk score, device trust score

    - 24h velocity, average amount, merchant diversity

    - Past fraud & disputes, merchant historical fraud rate

- Returns:

    - Fraud probability

    - Binary label (FRAUD / GENUINE)

    - Recommended action (ALLOW / SOFT_REVIEW / OTP_CHALLENGE / HARD_BLOCK)

- Uses the `FraudScorer` wrapper around saved model + encoders.

3. **Analytics & Plots**

- Class distribution (fraud vs genuine)

- Amount distributions by label

- Fraud rate by payment method and merchant category

- Temporal patterns (hour of day, weekday vs weekend)

- Behavioural & risk signals:

    - IP risk vs device trust (scatter plot)

    - Transaction count last 24h vs fraud

Each chart includes a short explanation to help stakeholders interpret the pattern.

4. **Project Pipeline**

- Slide‑free architecture walkthrough of the system:

    1. Data ingestion & validation

    2. Feature engineering & behavioural signals

    3. Imbalance handling & model training

    4. Model serialization & versioned artifacts

    5. Real‑time scoring & decisioning

    6. Testing, CI/CD & Streamlit UI

- Each step is shown as a styled card with:

    - What happens

    - Which files/modules are involved

    - The key question that step answers (e.g. “Is the data reliable enough to take risk decisions on?”)

## ⚙️ Configuration
Project settings are stored in `config/config.yaml`, e.g.:

You can tune:

- **Paths** to raw/interim/processed data

- **Model config** (algorithm, train/test split)

- **Decision threshold** (fraud_cutoff) that maps probabilities to labels/actions

## 🧠 Feature Engineering Highlights
Some of the key engineered features:

- **Velocity & activity**

    - Approximate `velocity_1h`, `velocity_24h`, `velocity_7d` from `txn_count_last_24h`

    - Transaction count bursts as indicators of bots or account takeover

- **Customer behaviour**

    - `amount_deviation`: difference between transaction amount and customer’s typical average

    - `transaction_success_rate_customer` proxy from OTP success rate

- **Device & network risk**

    - Device‑customer sharing count (how many customers use the same device)

    - IP address risk score

    - Device trust score

- **Historical fraud patterns**

    - Customer‑level and merchant‑level fraud/dispute indicators

    - Synthetic `historical_fraud_rate` derived from past fraud + disputes

- **Combined risk index**

    - Weighted combination of IP risk, device trust, customer history, and merchant history

## 🔍 Model & Evaluation
- **Model**: Tree‑based classifier (e.g., RandomForest) on engineered feature set

- **Imbalance handling**: SMOTE (oversampling minority fraud class)

- **Metrics**:

    - ROC AUC & PR AUC for ranking performance

    - Precision & recall at business‑aligned threshold

- **Threshold & actions**

## 🧪 CI/CD
The project is structured to work with GitHub Actions:

- On each push:

    - Install dependencies

    - Run `pytest`

- Optionally:

    - Run `python -m src.pipeline.run_pipeline` on new data

    - Trigger a redeploy of the Streamlit app

Add a `.github/workflows/ci.yml`

## 📌 Future Improvements
- Model explainability (SHAP/feature attribution in the app)

- Drift monitoring & automatic retraining

- Multiple model candidates (baseline vs advanced)

- API layer around FraudScorer (FastAPI / Flask) for integration with other systems

## 📄 License
MIT License

## 🙌 Acknowledgements
- Ideas inspired by public fraud detection examples and SMOTE‑based pipelines.

- Built with:

    - Streamlit for the UI

    - scikit‑learn

    - imbalanced‑learn / SMOTE

    - pytest for testing