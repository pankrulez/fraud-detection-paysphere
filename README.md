# 🛡️ PaySphere: Enterprise Risk Intelligence Microservice

![Pytest Status](https://github.com/pankrulez/fraud-detection-paysphere/actions/workflows/main.yml/badge.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Render](https://img.shields.io/badge/Render-Deployed-430098?logo=render&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?logo=streamlit&logoColor=white)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)

An asynchronous, decoupled machine learning microservice architecture designed to detect high-frequency payment fraud (UPI, Cards, Wallets) using **Vectorized Batch Inference** and **Explainable AI (XAI)**.

---

## 🏗️ System Architecture: The Decoupled Approach

PaySphere is architected as a **distributed system** to mimic a mission-critical fintech environment. 



### 1. The Brain (Inference API)
A **FastAPI** service hosted on **Render**. It serves a unified Scikit-Learn pipeline, handling:
- **Vectorized Batching:** A dedicated endpoint that reduced dashboard loading from **45s to 1.8s** by utilizing NumPy-backed vectorized scoring.
- **Pydantic Data Contracts:** Strict schema validation on 22 feature vectors to ensure 0% training-serving skew.

### 2. The Face (Risk Intelligence Console)
A **Streamlit** dashboard designed with a "Command Center" aesthetic. It provides:
- **Financial ROI Simulation:** Real-time analysis of "Fraud Saved" vs. "Customer Friction" costs.
- **Risk Fingerprinting:** Radar-based behavioral diagnostics and Waterfall XAI (Explainable AI).

---

## 🖥️ Operational Terminal Views

1. **🛡️ Executive Command Center:** High-level system health metrics and high-contrast (Turbo) risk concentration heatmaps.
2. **⚡ Real-Time Interceptor:** A diagnostic terminal for manual transaction overrides with real-time probability gauges and XAI.
3. **📂 Bulk Risk Assessment:** High-speed manifest scoring (up to 5,000 txns/chunk) with data-exfiltration (CSV export) capabilities.
4. **📊 Intelligence & ROI Simulator:** Strategic impact modeling. It visualizes the trade-off between security (Fraud Capture) and revenue (False Positives).
5. **⚙️ MLOps & Model Registry:** Technical lineage of the production artifact, metadata tracking, and live system audit logs.

---

## 🧠 Technical Performance & Engineering

- **Imbalance Mastery:** Utilized **SMOTE** (Synthetic Minority Over-sampling Technique) to handle the 0.5% fraud minority class, prioritizing **Precision-Recall AUC** over traditional accuracy.
- **Leakage-Proof Engineering:** Feature engineering for velocity (`txn_count_last_24h`) uses expanding windows to prevent temporal data leakage.
- **Explainable AI (XAI):** Implemented marginal feature substitution via the API proxy to provide a waterfall breakdown of mathematical "Decision Drivers."

---

## 🧱 Project Structure

```text
fraud-detection-paysphere/
├── app/
│   ├── main.py                 # Main Streamlit router
│   ├── overview_view.py        # Executive Dashboard
│   ├── live_view.py            # Real-Time Interceptor (XAI)
│   ├── batch_view.py           # Bulk Assessment (Uploader)
│   ├── analytics_view.py       # ROI Simulator & Intelligence
│   ├── pipeline_view.py        # MLOps Registry & Architecture
│   ├── ui_components.py        # Custom HTML/CSS component library
│   └── api.py                  # FastAPI Endpoints
```

---

## 📦 Run It Locally
1️⃣ Clone & Install
```bash
git clone [https://github.com/pankrulez/fraud-detection-paysphere.git](https://github.com/pankrulez/fraud-detection-paysphere.git)

pip install -r requirements.txt
```
2️⃣ Start the Backend (The Brain)

```bash
python -m uvicorn app.api:app --reload
```
3️⃣ Start the Frontend (The UI)

```bash
streamlit run app/main.py
```

---

## 📄 License
MIT License. Created by Pankaj Kapri.
