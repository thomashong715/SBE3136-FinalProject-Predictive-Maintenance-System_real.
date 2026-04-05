# FM Equipment Failure Prediction Agent
### Streamlit + Claude AI · Industrial Machinery Predictive Maintenance

---

## Setup (2 minutes)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key
```bash
# Mac/Linux
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows (Command Prompt)
set ANTHROPIC_API_KEY=sk-ant-...

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

Get your key at: https://console.anthropic.com

### 3. Run the app
```bash
streamlit run fm_failure_agent.py
```

App opens at: http://localhost:8501

---

## Live Demo Flow

1. **Sidebar → "Load demo data"** → click "Run Failure Prediction"
2. See the fleet overview KPIs (Critical / High / Medium counts)
3. Explore the Failure Probability bar chart and Risk Driver Heatmap
4. Click suggested questions in the chat — or type your own
5. Switch to **Upload CSV** and load your real cleaned data

---

## CSV Format (your data)

The agent adapts to any CSV with equipment sensor readings.
Ideal columns (any subset works):

| Column | Description |
|--------|-------------|
| equipment_id | Unique asset ID |
| equipment_type | Motor / Pump / Compressor etc. |
| location | Physical location |
| temperature_c | Operating temperature |
| vibration_mms | Vibration level (mm/s) |
| pressure_bar | System pressure |
| current_amps | Current draw |
| rpm | Rotational speed |
| oil_viscosity | Oil condition |
| last_maintenance_days | Days since last service |
| failure_history | Number of past failures |
| runtime_hours | Total operating hours |
| age_years | Equipment age |

---

## Agent Outputs per Asset

- **Failure probability** — 0–100% score
- **Days to failure** — estimated time to breakdown
- **Urgency alert** — Critical / High / Medium
- **Key indicators** — top contributing sensor readings
- **Maintenance recommendation** — specific actionable advice
- **Risk driver heatmap** — Temperature / Vibration / Maintenance / Age / Operational
- **AI chat** — ask any question about your fleet in plain English
