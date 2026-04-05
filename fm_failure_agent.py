import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anthropic
import json
import io
import time
from datetime import datetime, timedelta

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FM Equipment Failure Prediction Agent",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }

    /* Metric cards */
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #6b7280;
        margin-bottom: 8px;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-value {
        font-size: 36px;
        font-weight: 600;
        line-height: 1;
    }
    .metric-sub { font-size: 12px; color: #6b7280; margin-top: 6px; }

    .critical-val { color: #ef4444; }
    .high-val     { color: #f59e0b; }
    .medium-val   { color: #10b981; }
    .info-val     { color: #60a5fa; }

    /* Equipment cards */
    .eq-card {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 10px;
        transition: border-color 0.2s;
    }
    .eq-card:hover { border-color: #4a4d6e; }
    .eq-header { display: flex; align-items: center; justify-content: space-between; }
    .eq-id { font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: #9ca3af; }
    .eq-name { font-size: 15px; font-weight: 500; color: #f3f4f6; margin: 2px 0; }
    .eq-loc { font-size: 12px; color: #6b7280; }

    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
    }
    .badge-critical { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
    .badge-high     { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
    .badge-medium   { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }

    /* Progress bar */
    .prob-bar-bg { background: #2a2d3e; border-radius: 3px; height: 6px; overflow: hidden; margin-top: 8px; }
    .prob-bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s; }

    /* Header */
    .agent-header {
        background: linear-gradient(135deg, #1a1d27 0%, #0f1117 100%);
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 16px;
    }

    /* Chat */
    .chat-msg-user {
        background: #1e3a5f;
        border-radius: 10px 10px 2px 10px;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 14px;
        color: #e0f2fe;
        text-align: right;
    }
    .chat-msg-agent {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 10px 10px 10px 2px;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 14px;
        color: #d1d5db;
    }

    /* Sidebar */
    .sidebar-section {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .sidebar-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #6b7280;
        margin-bottom: 8px;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Divider */
    hr { border-color: #2a2d3e !important; }

    /* Override Streamlit defaults */
    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
        padding: 10px 24px;
        width: 100%;
    }
    .stButton > button:hover { background: #2563eb; }

    .stTextInput > div > div > input {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        color: #f3f4f6;
        border-radius: 8px;
    }

    [data-testid="stSidebar"] {
        background-color: #0f1117;
        border-right: 1px solid #2a2d3e;
    }

    .status-online {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: #10b981;
        font-family: 'IBM Plex Mono', monospace;
    }

    .section-title {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #6b7280;
        font-family: 'IBM Plex Mono', monospace;
        margin: 20px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #2a2d3e;
    }
</style>
""", unsafe_allow_html=True)

# ── Sample data ───────────────────────────────────────────────────────────────
SAMPLE_DATA = {
    "equipment_id":        ["EQ-001","EQ-002","EQ-003","EQ-004","EQ-005","EQ-006","EQ-007","EQ-008","EQ-009","EQ-010"],
    "equipment_type":      ["Centrifugal Pump","AC Motor","Compressor","Hydraulic Pump","AC Motor","Compressor","Centrifugal Pump","Gear Pump","AC Motor","Compressor"],
    "location":            ["Building A - F2","Building A - F1","Plant Room B","Workshop C","Building D - Roof","Plant Room A","Building B - B1","Production Line 1","Building C - F3","Plant Room C"],
    "age_years":           [8.2, 3.1, 12.6, 5.4, 9.8, 2.3, 15.1, 6.7, 1.2, 7.9],
    "temperature_c":       [87.4, 62.1, 94.8, 71.3, 78.6, 58.4, 101.2, 69.8, 55.2, 83.7],
    "vibration_mms":       [6.8, 2.1, 9.4, 3.9, 5.3, 1.4, 12.7, 4.2, 1.1, 7.6],
    "pressure_bar":        [4.2, None, 8.7, 5.1, None, 6.2, 3.8, 4.9, None, 7.4],
    "current_amps":        [18.5, 12.2, 31.4, 21.8, 16.9, 24.1, 22.4, 19.7, 11.8, 28.9],
    "oil_viscosity":       [42, None, 28, 38, None, 52, 19, 41, None, 33],
    "rpm":                 [1480, 2960, 740, 1470, 1480, 750, 1460, 960, 2960, 745],
    "last_maintenance_days":[180, 45, 320, 90, 210, 30, 410, 120, 15, 155],
    "failure_history":     [3, 0, 5, 1, 2, 0, 7, 2, 0, 3],
    "runtime_hours":       [14200, 5800, 22400, 9200, 17800, 3900, 28600, 11200, 2100, 13900],
}

# ── Session state ─────────────────────────────────────────────────────────────
if "predictions"    not in st.session_state: st.session_state.predictions    = None
if "df"             not in st.session_state: st.session_state.df             = None
if "chat_history"   not in st.session_state: st.session_state.chat_history   = []
if "analysis_done"  not in st.session_state: st.session_state.analysis_done  = False

# ── Anthropic client ──────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    return anthropic.Anthropic()

client = get_client()

# ── AI prediction ─────────────────────────────────────────────────────────────
def run_prediction(df: pd.DataFrame) -> dict:
    sample = df.head(20).to_dict(orient="records")
    prompt = f"""You are an expert FM predictive maintenance AI agent. Analyze this industrial equipment sensor data and return structured failure predictions.

Data columns: {list(df.columns)}
Total equipment: {len(df)}
Sample records: {json.dumps(sample, indent=1, default=str)}

Scoring rules:
- Temperature: >85°C warning (+20 prob), >95°C critical (+35 prob)  
- Vibration: >6 mm/s warning (+20 prob), >9 mm/s critical (+35 prob)
- Last maintenance: >200 days (+15 prob), >350 days (+25 prob)
- Failure history: each past failure adds 5 prob points
- Age: >10 years (+10 prob), >13 years (+20 prob)
- Runtime hours: >20000h (+15 prob)
- Low oil viscosity (<25): +20 prob
- High current draw (>30A): +15 prob

Urgency: critical (prob>=75), high (prob 45-74), medium (prob<45)
Days to failure: estimate realistically. critical=7-30 days, high=30-90 days, medium=90-365 days. Use 999 if >365.

Return ONLY valid JSON, no markdown, no explanation:
{{
  "summary": {{
    "total": <int>,
    "critical": <int>,
    "high": <int>,
    "medium": <int>,
    "avg_failure_probability": <float 0-100>
  }},
  "predictions": [
    {{
      "equipment_id": "<id>",
      "equipment_type": "<type>",
      "location": "<location>",
      "failure_probability": <int 0-100>,
      "days_to_failure": <int>,
      "urgency": "critical|high|medium",
      "key_indicators": ["<indicator1>", "<indicator2>", "<indicator3>"],
      "recommendation": "<specific 1-2 sentence maintenance action>",
      "risk_drivers": {{"temperature": <0-10>, "vibration": <0-10>, "maintenance": <0-10>, "age": <0-10>, "operational": <0-10>}}
    }}
  ]
}}"""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text.strip()
    text = text.replace("```json","").replace("```","").strip()
    return json.loads(text)


def ask_agent(question: str, predictions: dict, df: pd.DataFrame) -> str:
    context = f"""You are an expert FM equipment failure prediction agent. You have just analyzed {len(df)} pieces of industrial equipment.

Prediction summary: {json.dumps(predictions['summary'])}
Equipment predictions (top details): {json.dumps(predictions['predictions'][:10])}

Answer the user's question concisely and professionally. Be specific, reference equipment IDs when relevant.
Question: {question}"""
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=600,
        messages=[{"role": "user", "content": context}]
    )
    return response.content[0].text.strip()

# ── Color helpers ─────────────────────────────────────────────────────────────
URGENCY_COLOR = {"critical": "#ef4444", "high": "#f59e0b", "medium": "#10b981"}
URGENCY_BG    = {"critical": "rgba(239,68,68,0.08)", "high": "rgba(245,158,11,0.08)", "medium": "rgba(16,185,129,0.08)"}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:12px 0 20px'>
        <div style='font-size:20px;font-weight:600;color:#f3f4f6'>⚙️ FM Predict</div>
        <div style='font-size:12px;color:#6b7280;margin-top:2px;font-family:IBM Plex Mono,monospace'>Equipment Failure Agent v1.0</div>
        <div style='margin-top:10px' class='status-online'>● AGENT ONLINE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Data Source</div>', unsafe_allow_html=True)

    data_source = st.radio("", ["Upload CSV", "Use demo data"], label_visibility="collapsed")

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.session_state.predictions = None
            st.session_state.analysis_done = False
    else:
        if st.button("Load demo data"):
            st.session_state.df = pd.DataFrame(SAMPLE_DATA)
            st.session_state.predictions = None
            st.session_state.analysis_done = False
            st.rerun()

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown(f"""
        <div class='sidebar-section' style='margin-top:12px'>
            <div class='sidebar-label'>Loaded dataset</div>
            <div style='color:#f3f4f6;font-size:14px;font-weight:500'>{len(df)} equipment records</div>
            <div style='color:#6b7280;font-size:12px;margin-top:2px'>{len(df.columns)} features</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        if st.button("🔍  Run Failure Prediction", use_container_width=True):
            with st.spinner("AI agent analysing equipment data…"):
                try:
                    result = run_prediction(df)
                    st.session_state.predictions = result
                    st.session_state.analysis_done = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    st.markdown("---")
    st.markdown("""
    <div class='sidebar-label'>Thresholds</div>
    <div style='font-size:12px;color:#6b7280;line-height:1.8'>
    🔴 Critical &nbsp;≥ 75% prob<br>
    🟡 High &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;45–74% prob<br>
    🟢 Medium &nbsp;< 45% prob
    </div>
    """, unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:28px 0 8px;display:flex;align-items:center;gap:14px'>
    <div>
        <h1 style='font-size:26px;font-weight:600;color:#f3f4f6;margin:0'>
            FM Equipment Failure Prediction Agent
        </h1>
        <p style='color:#6b7280;font-size:14px;margin:4px 0 0'>
            Industrial machinery · Motors · Pumps · Compressors &nbsp;|&nbsp; Powered by Claude AI
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── No data state ─────────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    for col, icon, title, desc in [
        (col1, "📂", "Load your data", "Upload a cleaned CSV with equipment sensor readings, or use the built-in demo dataset"),
        (col2, "🤖", "AI analyses assets", "Claude reads every sensor channel — temperature, vibration, pressure, current, runtime — and scores each machine"),
        (col3, "📊", "Get predictions", "Failure probability, days to failure, and urgency alert per asset, with maintenance recommendations"),
    ]:
        with col:
            st.markdown(f"""
            <div style='background:#1a1d27;border:1px solid #2a2d3e;border-radius:10px;padding:20px;text-align:center;height:160px'>
                <div style='font-size:28px;margin-bottom:10px'>{icon}</div>
                <div style='font-size:14px;font-weight:500;color:#f3f4f6;margin-bottom:6px'>{title}</div>
                <div style='font-size:12px;color:#6b7280;line-height:1.5'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;margin-top:32px;color:#6b7280;font-size:13px'>
        ← Load data from the sidebar to get started
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.df

# ── Data preview (before analysis) ───────────────────────────────────────────
if not st.session_state.analysis_done:
    st.markdown('<div class="section-title">📋 Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=280)
    st.info("Click **Run Failure Prediction** in the sidebar to start the AI analysis.")
    st.stop()

# ── RESULTS ───────────────────────────────────────────────────────────────────
pred = st.session_state.predictions
preds = pred["predictions"]
summary = pred["summary"]

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Fleet Overview</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)

for col, label, value, cls in [
    (k1, "Total Assets",        summary["total"],              "info-val"),
    (k2, "Critical Alerts",     summary["critical"],           "critical-val"),
    (k3, "High Risk",           summary["high"],               "high-val"),
    (k4, "Monitored",           summary["medium"],             "medium-val"),
    (k5, "Avg Failure Prob",    f"{summary['avg_failure_probability']:.0f}%", "info-val"),
]:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value {cls}'>{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts row ────────────────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns([3, 2])

with chart_col1:
    st.markdown('<div class="section-title">🎯 Failure Probability by Asset</div>', unsafe_allow_html=True)
    sorted_preds = sorted(preds, key=lambda x: x["failure_probability"], reverse=True)
    ids   = [p["equipment_id"] for p in sorted_preds]
    probs = [p["failure_probability"] for p in sorted_preds]
    colors = [URGENCY_COLOR[p["urgency"]] for p in sorted_preds]

    fig = go.Figure(go.Bar(
        x=ids, y=probs,
        marker_color=colors,
        text=[f"{p}%" for p in probs],
        textposition="outside",
        textfont=dict(size=11, color="#9ca3af"),
    ))
    fig.add_hline(y=75, line_dash="dot", line_color="#ef4444", annotation_text="Critical threshold",
                  annotation_font_color="#ef4444", annotation_font_size=11)
    fig.add_hline(y=45, line_dash="dot", line_color="#f59e0b", annotation_text="High threshold",
                  annotation_font_color="#f59e0b", annotation_font_size=11)
    fig.update_layout(
        plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
        font=dict(color="#9ca3af", family="IBM Plex Sans"),
        xaxis=dict(gridcolor="#1a1d27", tickfont=dict(size=11)),
        yaxis=dict(gridcolor="#2a2d3e", range=[0, 115], title="Failure Probability (%)", titlefont=dict(size=12)),
        showlegend=False, margin=dict(l=40, r=20, t=20, b=40), height=280,
    )
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    st.markdown('<div class="section-title">⏱ Days to Failure Distribution</div>', unsafe_allow_html=True)
    dtf_data = []
    for p in preds:
        dtf = p["days_to_failure"]
        bucket = ">365 days" if dtf >= 999 else ("0–30 days" if dtf <= 30 else ("31–90 days" if dtf <= 90 else "91–365 days"))
        dtf_data.append({"bucket": bucket, "urgency": p["urgency"]})

    buckets = ["0–30 days", "31–90 days", "91–365 days", ">365 days"]
    counts  = [sum(1 for d in dtf_data if d["bucket"] == b) for b in buckets]
    bcolors = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6"]

    fig2 = go.Figure(go.Pie(
        labels=buckets, values=counts, hole=0.55,
        marker=dict(colors=bcolors, line=dict(color="#0f1117", width=2)),
        textfont=dict(size=11, color="white"),
    ))
    fig2.update_layout(
        plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
        font=dict(color="#9ca3af", family="IBM Plex Sans"),
        showlegend=True,
        legend=dict(font=dict(size=11), bgcolor="#0f1117"),
        margin=dict(l=10, r=10, t=10, b=10), height=280,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Risk radar heatmap ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔥 Risk Driver Heatmap</div>', unsafe_allow_html=True)

drivers = ["temperature", "vibration", "maintenance", "age", "operational"]
eq_ids  = [p["equipment_id"] for p in sorted_preds[:10]]
matrix  = [[p["risk_drivers"].get(d, 0) for d in drivers] for p in sorted_preds[:10]]

fig3 = go.Figure(go.Heatmap(
    z=matrix, x=[d.capitalize() for d in drivers], y=eq_ids,
    colorscale=[[0,"#1a1d27"],[0.3,"#1e3a5f"],[0.6,"#92400e"],[1,"#7f1d1d"]],
    showscale=True, colorbar=dict(
        title="Risk Level", thickness=12, len=0.8,
        tickfont=dict(color="#9ca3af", size=10),
        titlefont=dict(color="#9ca3af", size=11),
    ),
    text=[[str(v) for v in row] for row in matrix],
    texttemplate="%{text}", textfont=dict(size=11, color="white"),
))
fig3.update_layout(
    plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
    font=dict(color="#9ca3af", family="IBM Plex Sans"),
    xaxis=dict(side="top", tickfont=dict(size=12)),
    yaxis=dict(tickfont=dict(size=11)),
    margin=dict(l=80, r=40, t=40, b=20), height=320,
)
st.plotly_chart(fig3, use_container_width=True)

# ── Equipment cards ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔧 Equipment Predictions</div>', unsafe_allow_html=True)

urgency_filter = st.multiselect(
    "Filter by urgency",
    ["critical", "high", "medium"],
    default=["critical", "high", "medium"],
    label_visibility="collapsed",
    key="uf",
)

filtered = [p for p in sorted_preds if p["urgency"] in urgency_filter]

for p in filtered:
    urg = p["urgency"]
    dtf = "≥ 365 days" if p["days_to_failure"] >= 999 else f"{p['days_to_failure']} days"
    col_l, col_r = st.columns([3, 1])
    with col_l:
        st.markdown(f"""
        <div class='eq-card' style='border-left:3px solid {URGENCY_COLOR[urg]}'>
            <div style='display:flex;align-items:flex-start;justify-content:space-between'>
                <div>
                    <div class='eq-id'>{p['equipment_id']}</div>
                    <div class='eq-name'>{p['equipment_type']}</div>
                    <div class='eq-loc'>📍 {p['location']}</div>
                </div>
                <span class='badge badge-{urg}'>{urg.upper()}</span>
            </div>
            <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:12px'>
                <div>
                    <div style='font-size:11px;color:#6b7280;font-family:IBM Plex Mono,monospace;text-transform:uppercase;letter-spacing:.5px'>Failure Prob.</div>
                    <div style='font-size:20px;font-weight:600;color:{URGENCY_COLOR[urg]}'>{p['failure_probability']}%</div>
                    <div class='prob-bar-bg'><div class='prob-bar-fill' style='width:{p["failure_probability"]}%;background:{URGENCY_COLOR[urg]}'></div></div>
                </div>
                <div>
                    <div style='font-size:11px;color:#6b7280;font-family:IBM Plex Mono,monospace;text-transform:uppercase;letter-spacing:.5px'>Days to Failure</div>
                    <div style='font-size:20px;font-weight:600;color:#f3f4f6'>{dtf}</div>
                </div>
                <div>
                    <div style='font-size:11px;color:#6b7280;font-family:IBM Plex Mono,monospace;text-transform:uppercase;letter-spacing:.5px'>Key Indicators</div>
                    <div style='font-size:12px;color:#d1d5db;margin-top:4px'>{"  ·  ".join(p["key_indicators"][:2])}</div>
                </div>
            </div>
            <div style='margin-top:12px;padding:10px 12px;background:#111318;border-radius:6px;font-size:13px;color:#9ca3af;border-left:2px solid {URGENCY_COLOR[urg]}'>
                <strong style='color:#d1d5db'>Recommendation:</strong> {p['recommendation']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Data table ────────────────────────────────────────────────────────────────
with st.expander("📋 View raw prediction table"):
    table_df = pd.DataFrame([{
        "ID": p["equipment_id"],
        "Type": p["equipment_type"],
        "Location": p["location"],
        "Failure Prob %": p["failure_probability"],
        "Days to Failure": "365+" if p["days_to_failure"] >= 999 else p["days_to_failure"],
        "Urgency": p["urgency"].upper(),
        "Recommendation": p["recommendation"][:80] + "…"
    } for p in sorted_preds])
    st.dataframe(table_df, use_container_width=True, hide_index=True)
    csv_buf = io.StringIO()
    table_df.to_csv(csv_buf, index=False)
    st.download_button("⬇ Download predictions CSV", csv_buf.getvalue(), "fm_predictions.csv", "text/csv")

# ── Agent chat ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">💬 Ask the Agent</div>', unsafe_allow_html=True)

chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-msg-user'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-msg-agent'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

q_col, btn_col = st.columns([5, 1])
with q_col:
    user_q = st.text_input("", placeholder="e.g. Which pump should I prioritise? What's causing EQ-003 to be critical?", label_visibility="collapsed", key="chat_input")
with btn_col:
    ask_clicked = st.button("Ask →", use_container_width=True)

if ask_clicked and user_q.strip():
    st.session_state.chat_history.append({"role": "user", "content": user_q})
    with st.spinner("Agent thinking…"):
        answer = ask_agent(user_q, pred, df)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.rerun()

# ── Suggested questions ───────────────────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown("""
    <div style='margin-top:8px;display:flex;gap:8px;flex-wrap:wrap'>
        <div style='font-size:11px;color:#6b7280;align-self:center'>Suggested:</div>
    </div>
    """, unsafe_allow_html=True)
    sq_cols = st.columns(3)
    suggestions = [
        "Which equipment needs urgent attention this week?",
        "What are the top 3 root causes driving failures?",
        "Give me a maintenance schedule for critical assets",
    ]
    for col, suggestion in zip(sq_cols, suggestions):
        with col:
            if st.button(suggestion, key=f"sq_{suggestion[:20]}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": suggestion})
                with st.spinner("Agent thinking…"):
                    answer = ask_agent(suggestion, pred, df)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()
