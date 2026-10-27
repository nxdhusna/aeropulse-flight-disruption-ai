# ✈️ AeroPulse AI: Intelligent Flight Disruption & Turnaround Recovery Platform

An enterprise-grade **Airport Operations Control Center (AOCC) Intelligence System** built with **Dual-Stage XGBoost (Classification + Regression)**, **SHAP Explainable AI**, and **Streamlit**. 

Designed to help airlines and hub operators monitor On-Time Performance (OTP), forecast flight delay probabilities and exact delay durations in minutes, simulate aircraft tail-rotation knock-on cascades, and recommend prescriptive ground-handling interventions.

---

## 🚀 Key Features

* **📡 Live Geospatial AOCC Radar Map:** Interactive Plotly Mapbox visualization of ASEAN and international route corridors radiating out of the KLIA Hub (`KUL`).
* **⚡ Dual-Stage ML Prediction Engine:** 
  * *Stage-1 Classifier:* Predicts the probability of an official OTP delay breach ($\ge 15$ mins) with **0.9851 ROC-AUC**.
  * *Stage-2 Regressor:* Predicts exact delay duration in minutes (**±3.84m MAE**).
* **🔍 SHAP Explainable AI (Root-Cause Attribution):** Decomposes black-box predictions to quantify the exact minute-by-minute impact of inbound aircraft rotation, weather severity, and airport slot congestion.
* **⛓️ Aircraft Tail-Rotation ("Domino Effect") Tracker:** Visualizes how an inbound delay cascades across multi-leg daily schedules for specific aircraft tail numbers (e.g., `9M-MNA`).
* **🛠️ Turnaround Recovery Simulator:** Interactive AOCC control panel allowing flight dispatchers to simulate ground interventions (rapid cleaning, dual-bridge boarding, priority taxi slots) and instantly measure delay mitigation.
* **📊 Power BI Hub Telemetry Pipeline:** Structured operational dataset (`klia_flight_disruptions_powerbi.csv`) ready for executive Power BI dashboarding.

---

## 📊 Model Performance & Dataset

Trained and validated on **20,000 real commercial flight records** (*US DOT / Bureau of Transportation Statistics Benchmark*) enriched with AOCC turnaround physics and airport congestion telemetry:

| Metric | Validation Score |
| :--- | :--- |
| **Stage-1 Classifier ROC-AUC** | **0.9851** |
| **Stage-1 Accuracy** | **94.70%** |
| **Stage-2 Regressor $R^2$ Score** | **0.9482** |
| **Stage-2 Mean Absolute Error (MAE)** | **±3.84 minutes** |

---

## 🛠️ Tech Stack & Architecture

* **Machine Learning:** `XGBoost`, `Scikit-Learn`, `Pandas`, `NumPy`
* **Explainable AI (XAI):** `SHAP (SHapley Additive exPlanations)`
* **Frontend & GIS:** `Streamlit`, `Plotly`, `Plotly Mapbox`
* **Business Intelligence:** `Power BI Pipeline` (CSV export schema)

```text
aeropulse-flight-disruption-ai/
├── data/
│   ├── real_bts_flights_20k.csv           # 20,000 real US DOT/BTS flight records
│   └── klia_flight_disruptions_powerbi.csv  # Enriched hub telemetry for Power BI
├── models/
│   ├── xgb_delay_classifier.pkl           # Stage-1 OTP Risk Classifier (0.9851 AUC)
│   ├── xgb_delay_regressor.pkl            # Stage-2 Delay Duration Regressor (±3.84m MAE)
│   └── shap_flight_explainer.pkl          # Serialized SHAP TreeExplainer
├── src/
│   ├── train_pipeline.py                  # Real data downloader, cleaning, & dual-model training
│   └── disruption_engine.py               # Inference, SHAP attribution, & tail-rotation cascade logic
├── app.py                                 # Interactive Streamlit AOCC Control Center UI
└── requirements.txt

# 1. Clone the repository
git clone [https://github.com/nxdhusna/aeropulse-flight-disruption-ai.git](https://github.com/nxdhusna/aeropulse-flight-disruption-ai.git)
cd aeropulse-flight-disruption-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models & generate SHAP explainer (optional - pre-trained models included)
python src/train_pipeline.py

# 4. Launch the live AOCC web dashboard
streamlit run app.py