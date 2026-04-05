"""
FM Equipment Failure Prediction Agent
======================================
Run with:
    pip install streamlit anthropic pandas scikit-learn plotly
    streamlit run fm_failure_prediction_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from anthropic import Anthropic
import json
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FM Failure Prediction Agent",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main { background-color: #0f1117; }

    /* Metric cards */
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .label {
        font-size: 11px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Risk gauge colours */
    .risk-critical { color: #ef4444; }
    .risk-high     { color: #f97316; }
    .risk-medium   { color: #eab308; }
    .risk-low      { color: #22c55e; }

    /* Section headers */
    .section-title {
        font-size: 13px;
        font-weight: 500;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 12px;
        border-bottom: 1px solid #2a2d3e;
        padding-bottom: 6px;
    }

    /* Factor pills */
    .factor-pill {
        display: inline-block;
        background: #1e2030;
        border: 1px solid #3a3d52;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 12px;
        color: #a0aec0;
        margin: 3px;
    }

    /* Result banners */
    .banner-failure {
        background: linear-gradient(135deg, #2d1515, #3d1f1f);
        border: 1px solid #7f1d1d;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        color: #fca5a5;
    }
    .banner-ok {
        background: linear-gradient(135deg, #0f2d1e, #142b20);
        border: 1px solid #14532d;
        border-left: 4px solid #22c55e;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        color: #86efac;
    }
    .banner-title { font-size: 16px; font-weight: 600; margin-bottom: 6px; }
    .banner-body  { font-size: 13px; line-height: 1.7; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0d0f18;
        border-right: 1px solid #1f2234;
    }
    [data-testid="stSidebar"] .stMarkdown { color: #9ca3af; }

    /* Streamlit overrides */
    .stButton>button {
        background: #1d4ed8;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
        transition: background 0.2s;
    }
    .stButton>button:hover { background: #1e40af; border: none; }
    .stButton>button[kind="secondary"] {
        background: #1a1d27;
        color: #a0aec0;
        border: 1px solid #2a2d3e;
    }
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        color: #e2e8f0;
        border-radius: 8px;
    }
    div[data-testid="stFileUploader"] {
        background: #1a1d27;
        border: 2px dashed #2a2d3e;
        border-radius: 10px;
    }
    .stDataFrame { background: #1a1d27; }
    .stAlert { border-radius: 8px; }
    .stSpinner > div { border-top-color: #1d4ed8 !important; }
    h1 { font-weight: 600 !important; }
    h2 { font-weight: 500 !important; font-size: 1.1rem !important; color: #e2e8f0 !important; }
    h3 { font-weight: 500 !important; color: #9ca3af !important; font-size: 0.95rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Anthropic client ───────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    return Anthropic(api_key=api_key)


# ── Helper: compute dataset statistics ────────────────────────────────────────
def compute_stats(df):
    rpm_col   = "Rotational speed [rpm]"
    torq_col  = "Torque "
    wear_col  = "Tool wear "
    air_col   = "Air temperature [C]"
    proc_col  = "Process temperature [C]"
    fail_col  = "Machine failure"
    type_col  = "Type"

    total    = len(df)
    failures = int(df[fail_col].sum())
    fail_rate = round(failures / total * 100, 2) if total else 0

    fail_by_type = df[df[fail_col] == 1][type_col].value_counts().to_dict()

    high_wear_fail = int(df[(df[wear_col] > 200) & (df[fail_col] == 1)].shape[0])

    return {
        "total": total,
        "failures": failures,
        "fail_rate": fail_rate,
        "avg_rpm": round(df[rpm_col].mean(), 1),
        "min_rpm": int(df[rpm_col].min()),
        "max_rpm": int(df[rpm_col].max()),
        "avg_torque": round(df[torq_col].mean(), 1),
        "avg_wear": round(df[wear_col].mean(), 1),
        "max_wear": round(df[wear_col].max(), 1),
        "avg_air_temp": round(df[air_col].mean(), 2),
        "avg_proc_temp": round(df[proc_col].mean(), 2),
        "fail_by_type": fail_by_type,
        "high_wear_failures": high_wear_fail,
        "types": sorted(df[type_col].dropna().unique().tolist()),
    }


# ── Helper: AI dataset analysis ───────────────────────────────────────────────
def ai_dataset_analysis(client, stats, columns):
    prompt = f"""You are an expert FM (Facility Management) data analyst. Analyse this maintenance dataset and provide professional insights.

Dataset summary:
- Total records: {stats['total']:,}
- Machine failures: {stats['failures']} ({stats['fail_rate']}% failure rate)
- Machine types: {', '.join(stats['types'])}
- Columns: {', '.join(columns)}

Key statistics:
- Avg RPM: {stats['avg_rpm']} | Range: {stats['min_rpm']} – {stats['max_rpm']}
- Avg Torque: {stats['avg_torque']} Nm
- Avg Tool wear: {stats['avg_wear']} min | Max: {stats['max_wear']} min
- Avg Air temp: {stats['avg_air_temp']}°C | Avg Process temp: {stats['avg_proc_temp']}°C
- Failures by machine type: {stats['fail_by_type']}
- High-risk cases (tool wear > 200 min with failure): {stats['high_wear_failures']}

Write a professional dataset health assessment in 3 short paragraphs:
1. Overall fleet health and failure rate interpretation
2. Key risk factors and their significance
3. Which machine type and operating conditions are most at risk

Use FM domain language. No markdown headers. Plain paragraphs only."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


# ── Helper: AI failure prediction ─────────────────────────────────────────────
def ai_predict_failure(client, stats, inputs):
    temp_diff = round(inputs["proc_temp"] - inputs["air_temp"], 1)
    prompt = f"""You are an AI equipment failure prediction model for a Facility Management system.

Training data context:
- {stats['total']:,} records | {stats['fail_rate']}% overall failure rate
- Avg RPM: {stats['avg_rpm']} | Avg Torque: {stats['avg_torque']} Nm
- Avg tool wear: {stats['avg_wear']} min | Max observed: {stats['max_wear']} min
- Failure breakdown by type: {stats['fail_by_type']}

Current equipment sensor reading:
- Machine type: {inputs['machine_type']}
- Rotational speed: {inputs['rpm']} RPM
- Torque: {inputs['torque']} Nm
- Tool wear: {inputs['tool_wear']} min
- Air temperature: {inputs['air_temp']}°C
- Process temperature: {inputs['proc_temp']}°C
- Temperature differential: {temp_diff}°C

Apply threshold-based and statistical reasoning (as taught in ML courses) to predict failure. Compare readings to population averages and known failure thresholds.

Respond ONLY with a valid JSON object, no preamble or markdown:
{{
  "failure_predicted": true or false,
  "confidence": "High" or "Medium" or "Low",
  "risk_score": integer 0-100,
  "risk_level": "Critical" or "High" or "Medium" or "Low",
  "triggered_factors": ["factor 1", "factor 2"],
  "predicted_failure_mode": "Tool wear failure" or "Heat dissipation failure" or "Overstrain failure" or "Power failure" or "None",
  "explanation": "2-sentence explanation"
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip().replace("```json", "").replace("```", "")
    return json.loads(raw)


# ── Helper: AI maintenance recommendation ─────────────────────────────────────
def ai_recommendation(client, pred, inputs):
    prompt = f"""You are a senior FM (Facility Management) maintenance engineer. Write a structured maintenance recommendation based on this AI prediction.

Prediction result:
- Failure predicted: {pred['failure_predicted']}
- Risk level: {pred['risk_level']} | Risk score: {pred['risk_score']}/100
- Confidence: {pred['confidence']}
- Predicted failure mode: {pred['predicted_failure_mode']}
- Triggered risk factors: {', '.join(pred.get('triggered_factors', []))}
- Machine type: {inputs['machine_type']} | Tool wear: {inputs['tool_wear']} min

Provide a structured recommendation with these 4 sections:
1. **Immediate Action** — what to do right now (1-2 sentences)
2. **Maintenance Tasks** — 3 specific numbered tasks
3. **Monitoring Parameters** — 3 KPIs to watch with target thresholds
4. **Priority & Time Window** — urgency level and estimated window before failure if untreated

Use professional FM language. Use markdown formatting."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=700,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


# ── Helper: Plotly charts ──────────────────────────────────────────────────────
def chart_failure_by_type(df):
    counts = df.groupby(["Type", "Machine failure"]).size().reset_index(name="count")
    counts["status"] = counts["Machine failure"].map({0: "Normal", 1: "Failure"})
    fig = px.bar(
        counts, x="Type", y="count", color="status",
        color_discrete_map={"Normal": "#1d4ed8", "Failure": "#ef4444"},
        barmode="group", template="plotly_dark",
        labels={"count": "Count", "Type": "Machine Type"},
    )
    fig.update_layout(
        paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
        legend=dict(title="", font=dict(size=11)),
        margin=dict(t=20, b=20, l=10, r=10), height=260,
    )
    return fig


def chart_wear_distribution(df):
    fail = df[df["Machine failure"] == 1]["Tool wear "]
    ok   = df[df["Machine failure"] == 0]["Tool wear "]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ok,   name="Normal",  marker_color="#1d4ed8", opacity=0.7, nbinsx=40))
    fig.add_trace(go.Histogram(x=fail, name="Failure", marker_color="#ef4444", opacity=0.8, nbinsx=40))
    fig.update_layout(
        barmode="overlay", template="plotly_dark",
        paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
        legend=dict(font=dict(size=11)),
        margin=dict(t=20, b=20, l=10, r=10), height=260,
        xaxis_title="Tool wear (min)", yaxis_title="Count",
    )
    return fig


def chart_rpm_torque_scatter(df):
    sample = df.sample(min(1000, len(df)), random_state=42)
    sample["Status"] = sample["Machine failure"].map({0: "Normal", 1: "Failure"})
    fig = px.scatter(
        sample, x="Rotational speed [rpm]", y="Torque ",
        color="Status",
        color_discrete_map={"Normal": "#1d4ed8", "Failure": "#ef4444"},
        opacity=0.6, template="plotly_dark",
        labels={"Torque ": "Torque (Nm)"},
    )
    fig.update_layout(
        paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
        legend=dict(title="", font=dict(size=11)),
        margin=dict(t=20, b=20, l=10, r=10), height=260,
    )
    return fig


def chart_risk_gauge(score):
    color = "#ef4444" if score >= 75 else "#f97316" if score >= 50 else "#eab308" if score >= 25 else "#22c55e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Risk Score", "font": {"size": 13, "color": "#9ca3af"}},
        number={"font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4b5563", "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1a1d27",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25],   "color": "#0d3321"},
                {"range": [25, 50],  "color": "#2d2a0d"},
                {"range": [50, 75],  "color": "#2d1a0d"},
                {"range": [75, 100], "color": "#2d0d0d"},
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor="#1a1d27", height=230,
        margin=dict(t=30, b=10, l=20, r=20),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ FM Failure Prediction")
    st.markdown("---")

    api_key_input = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get your key at console.anthropic.com",
    )
    if api_key_input:
        st.session_state["api_key"] = api_key_input

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
1. Enter your API key above  
2. Upload your cleaned CSV  
3. Explore dataset insights  
4. Enter sensor readings  
5. Get AI failure prediction  
6. Review maintenance recommendation
""")
    st.markdown("---")
    st.markdown("### Expected columns")
    st.markdown("""
- `Type` (L / M / H)  
- `Rotational speed [rpm]`  
- `Torque `  
- `Tool wear `  
- `Machine failure` (0/1)  
- `Air temperature [C]`  
- `Process temperature [C]`  
- `Failure Mode`
""")
    st.markdown("---")
    st.caption("Built for FM professionals using the Anthropic API")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
st.title("Equipment Failure Prediction Agent")
st.caption("AI-powered predictive maintenance for facility management")

# Get client
api_key = st.session_state.get("api_key", "")
client = Anthropic(api_key=api_key) if api_key else None

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📂 Data Upload", "📊 Dataset Insights", "🔮 Predict Failure", "🔧 Recommendation"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Upload maintenance data")
    st.markdown("Upload your cleaned CSV file to begin. The agent will analyse the dataset and use it to calibrate failure predictions.")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.session_state["df"] = df

            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            # Re-add trailing space for torque/tool wear as in original
            rename_map = {}
            for col in df.columns:
                if "torque" in col.lower() and not col.endswith(" "):
                    rename_map[col] = col + " "
                if "tool wear" in col.lower() and not col.endswith(" "):
                    rename_map[col] = col + " "
            if rename_map:
                df.rename(columns=rename_map, inplace=True)

            stats = compute_stats(df)
            st.session_state["stats"] = stats

            st.success(f"✓ Loaded **{stats['total']:,}** records from `{uploaded.name}`")

            # Metric cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""<div class="metric-card"><div class="label">Total records</div>
                <div class="value" style="color:#60a5fa">{stats['total']:,}</div></div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class="metric-card"><div class="label">Failure events</div>
                <div class="value risk-high">{stats['failures']}</div></div>""", unsafe_allow_html=True)
            with col3:
                color = "risk-critical" if stats['fail_rate'] > 5 else "risk-medium" if stats['fail_rate'] > 2 else "risk-low"
                st.markdown(f"""<div class="metric-card"><div class="label">Failure rate</div>
                <div class="value {color}">{stats['fail_rate']}%</div></div>""", unsafe_allow_html=True)
            with col4:
                st.markdown(f"""<div class="metric-card"><div class="label">Machine types</div>
                <div class="value" style="color:#a78bfa">{', '.join(stats['types'])}</div></div>""", unsafe_allow_html=True)

            st.markdown("")
            st.markdown("#### Data preview (first 10 rows)")
            st.dataframe(df.head(10), use_container_width=True)

            st.markdown("#### Column summary")
            st.dataframe(df.describe().round(2), use_container_width=True)

        except Exception as e:
            st.error(f"Failed to read file: {e}")
    else:
        st.info("Please upload a CSV file to continue.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATASET INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if "df" not in st.session_state:
        st.info("Upload a dataset in the **Data Upload** tab first.")
    else:
        df    = st.session_state["df"]
        stats = st.session_state["stats"]

        st.markdown("### Dataset insights")

        # Charts row 1
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### Failures by machine type")
            st.plotly_chart(chart_failure_by_type(df), use_container_width=True)
        with col_b:
            st.markdown("##### Tool wear distribution")
            st.plotly_chart(chart_wear_distribution(df), use_container_width=True)

        # Charts row 2
        st.markdown("##### RPM vs Torque — failure overlay")
        st.plotly_chart(chart_rpm_torque_scatter(df), use_container_width=True)

        # AI analysis
        st.markdown("---")
        st.markdown("### AI dataset health assessment")
        if not client:
            st.warning("Enter your Anthropic API key in the sidebar to enable AI analysis.")
        else:
            if "dataset_analysis" not in st.session_state:
                if st.button("Run AI analysis", key="run_analysis"):
                    with st.spinner("Analysing dataset..."):
                        try:
                            analysis = ai_dataset_analysis(client, stats, list(df.columns))
                            st.session_state["dataset_analysis"] = analysis
                        except Exception as e:
                            st.error(f"AI analysis failed: {e}")

            if "dataset_analysis" in st.session_state:
                st.markdown(st.session_state["dataset_analysis"])
                if st.button("Re-run analysis", key="rerun_analysis"):
                    del st.session_state["dataset_analysis"]
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICT FAILURE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if "df" not in st.session_state:
        st.info("Upload a dataset in the **Data Upload** tab first.")
    else:
        stats = st.session_state["stats"]
        df    = st.session_state["df"]

        st.markdown("### Enter current equipment sensor readings")
        st.caption("The AI will compare these readings to the patterns in your uploaded dataset.")

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                machine_type = st.selectbox(
                    "Machine type",
                    options=["L", "M", "H"],
                    index=1,
                    help="L = Light duty, M = Medium duty, H = Heavy duty",
                )
                rpm = st.number_input(
                    "Rotational speed (RPM)",
                    min_value=0, max_value=5000,
                    value=1500, step=10,
                )

            with col2:
                torque = st.number_input(
                    "Torque (Nm)",
                    min_value=0.0, max_value=100.0,
                    value=40.0, step=0.5,
                )
                tool_wear = st.number_input(
                    "Tool wear (min)",
                    min_value=0, max_value=300,
                    value=50, step=1,
                )

            with col3:
                air_temp = st.number_input(
                    "Air temperature (°C)",
                    min_value=15.0, max_value=45.0,
                    value=25.0, step=0.1,
                )
                proc_temp = st.number_input(
                    "Process temperature (°C)",
                    min_value=25.0, max_value=55.0,
                    value=36.0, step=0.1,
                )

            submitted = st.form_submit_button("Run failure prediction", use_container_width=True)

        if submitted:
            inputs = {
                "machine_type": machine_type,
                "rpm": rpm,
                "torque": torque,
                "tool_wear": tool_wear,
                "air_temp": air_temp,
                "proc_temp": proc_temp,
            }
            st.session_state["last_inputs"] = inputs

            if not client:
                st.warning("Enter your Anthropic API key in the sidebar.")
            else:
                with st.spinner("Running AI failure prediction..."):
                    try:
                        pred = ai_predict_failure(client, stats, inputs)
                        st.session_state["last_prediction"] = pred
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

        # Show result
        if "last_prediction" in st.session_state:
            pred = st.session_state["last_prediction"]
            is_fail = pred.get("failure_predicted", False)

            st.markdown("---")
            st.markdown("### Prediction result")

            col_gauge, col_result = st.columns([1, 2])

            with col_gauge:
                st.plotly_chart(chart_risk_gauge(pred["risk_score"]), use_container_width=True)

            with col_result:
                level_colors = {
                    "Critical": "#ef4444", "High": "#f97316",
                    "Medium":   "#eab308", "Low":  "#22c55e",
                }
                badge_color = level_colors.get(pred["risk_level"], "#6b7280")

                if is_fail:
                    st.markdown(f"""
<div class="banner-failure">
  <div class="banner-title">⚠ Failure Predicted</div>
  <div class="banner-body">{pred.get('explanation', '')}</div>
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div class="banner-ok">
  <div class="banner-title">✓ No Failure Predicted</div>
  <div class="banner-body">{pred.get('explanation', '')}</div>
</div>""", unsafe_allow_html=True)

                st.markdown("")
                cols = st.columns(3)
                cols[0].metric("Risk level",  pred["risk_level"])
                cols[1].metric("Confidence",  pred["confidence"])
                cols[2].metric("Failure mode", pred.get("predicted_failure_mode", "None"))

                factors = pred.get("triggered_factors", [])
                if factors:
                    st.markdown("**Triggered risk factors:**")
                    pills = " ".join([f'<span class="factor-pill">{f}</span>' for f in factors])
                    st.markdown(pills, unsafe_allow_html=True)

            st.markdown("---")
            st.info("Go to the **Recommendation** tab to get a full maintenance action plan.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if "last_prediction" not in st.session_state:
        st.info("Run a prediction in the **Predict Failure** tab first.")
    else:
        pred   = st.session_state["last_prediction"]
        inputs = st.session_state.get("last_inputs", {})

        st.markdown("### Maintenance recommendation")

        # Summary banner
        is_fail = pred.get("failure_predicted", False)
        risk    = pred.get("risk_level", "")
        score   = pred.get("risk_score", 0)
        mode    = pred.get("predicted_failure_mode", "None")

        col1, col2, col3 = st.columns(3)
        col1.metric("Risk score",     f"{score}/100")
        col2.metric("Risk level",     risk)
        col3.metric("Predicted mode", mode)

        st.markdown("")

        if not client:
            st.warning("Enter your Anthropic API key in the sidebar.")
        else:
            if "recommendation" not in st.session_state or st.button("Regenerate recommendation"):
                with st.spinner("Generating maintenance recommendation..."):
                    try:
                        rec = ai_recommendation(client, pred, inputs)
                        st.session_state["recommendation"] = rec
                    except Exception as e:
                        st.error(f"Recommendation failed: {e}")

            if "recommendation" in st.session_state:
                st.markdown(st.session_state["recommendation"])

                st.markdown("---")
                # Export
                st.markdown("### Export report")
                report_text = f"""FM EQUIPMENT FAILURE PREDICTION REPORT
{'='*50}

PREDICTION SUMMARY
------------------
Failure predicted : {pred.get('failure_predicted')}
Risk score        : {score}/100
Risk level        : {risk}
Confidence        : {pred.get('confidence')}
Failure mode      : {mode}
Triggered factors : {', '.join(pred.get('triggered_factors', []))}

SENSOR READINGS
---------------
Machine type      : {inputs.get('machine_type')}
RPM               : {inputs.get('rpm')}
Torque            : {inputs.get('torque')} Nm
Tool wear         : {inputs.get('tool_wear')} min
Air temperature   : {inputs.get('air_temp')}°C
Process temp      : {inputs.get('proc_temp')}°C

AI EXPLANATION
--------------
{pred.get('explanation', '')}

MAINTENANCE RECOMMENDATION
--------------------------
{st.session_state['recommendation']}
"""
                st.download_button(
                    label="Download report (.txt)",
                    data=report_text,
                    file_name="fm_failure_prediction_report.txt",
                    mime="text/plain",
                )
