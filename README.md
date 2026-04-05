# FM AI Agent v2 — Equipment Failure Prediction

A full-stack AI agent for predictive maintenance of HVAC and Elevator equipment, built with Streamlit.

## Features

| Page | Capability |
|---|---|
| 📊 Dashboard | Live overview of all equipment — failure probabilities, risk distribution, model F1 scores |
| 🔮 Prediction | Sensor input sliders → ML model → failure probability + SHAP explainability + rule-based diagnosis |
| 🌀 Simulation | Stress-test equipment under 5 degradation modes over time (Gradual wear, Sudden spike, etc.) |
| 📡 Real-time | Live sensor simulation with streaming probability chart + event log |
| 🔧 Work Orders | AI-generated prioritized maintenance tasks with create/assign/close workflow |
| 💬 Feedback | Technician feedback form → SQLite DB → retraining readiness tracker |
| ⚙️ AutoML | Upload custom CSV → automated preprocess → multi-model train → evaluate → deploy |

## ML Models

- **AHU, Chiller, Elevator**: Calibrated Random Forest (scikit-learn) trained on synthetic sensor data
- **Evaluation**: Precision, Recall, F1-score via 5-fold cross-validation
- **Explainability**: Feature importance–weighted SHAP approximation
- **Rule engine**: Threshold-based diagnosis per equipment type
- **AutoML**: Compares Random Forest, Gradient Boosting, Logistic Regression on user-uploaded data

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

## Project Structure

```
fm_ai_agent/
├── app.py                  # Entry point + sidebar navigation
├── requirements.txt
├── models/                 # Auto-created — cached trained models (.pkl)
├── feedback.db             # Auto-created — SQLite feedback store
└── pages_code/
    ├── __init__.py
    ├── data.py             # Shared models, data generation, DB utils
    ├── dashboard.py        # Overview page
    ├── prediction.py       # Sensor input + ML prediction
    ├── simulation.py       # Degradation scenario simulation
    ├── realtime.py         # Live sensor streaming
    ├── workorders.py       # Work order management
    ├── feedback.py         # Feedback loop + analytics
    └── automl.py           # Custom dataset AutoML pipeline
```

## Extending

- **Add equipment type**: Add entries to `FEATURES`, `RULES` in `data.py`
- **Connect BMS/IoT**: Replace synthetic sensor generation in `realtime.py` with MQTT/REST API calls
- **Vertex AI deployment**: Replace `joblib` model cache in `data.py` with Vertex AI Endpoint calls
- **Real SHAP**: Replace approximation in `data.py:predict()` with `shap.TreeExplainer`
