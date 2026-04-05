import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pages_code.data import EQUIPMENT, FEATURES, get_model, predict, risk_label, risk_color


def show():
    st.title("📊 Dashboard")
    st.caption("FM AI Agent v2 · Equipment Failure Prediction Overview")

    # ── Pre-compute probabilities for all equipment ───────────────────────────
    @st.cache_data(ttl=60)
    def get_all_probs():
        results = []
        np.random.seed(99)
        defaults = {
            "AHU-01": {"supply_air_temp":34,"return_air_temp":38,"filter_dp":310,
                       "fan_vibration":11,"coil_fouling":0.75,"runtime_hrs":7800,"ambient_humidity":70},
            "AHU-02": {"supply_air_temp":29,"return_air_temp":33,"filter_dp":220,
                       "fan_vibration":7,"coil_fouling":0.55,"runtime_hrs":5200,"ambient_humidity":60},
            "AHU-03": {"supply_air_temp":22,"return_air_temp":26,"filter_dp":120,
                       "fan_vibration":3,"coil_fouling":0.20,"runtime_hrs":2100,"ambient_humidity":50},
            "CHR-01": {"condenser_temp":46,"evaporator_temp":8,"refrigerant_pres":20,
                       "cop":2.8,"compressor_vib":8.5,"runtime_hrs":8200},
            "CHR-02": {"condenser_temp":36,"evaporator_temp":6,"refrigerant_pres":14,
                       "cop":4.2,"compressor_vib":2.5,"runtime_hrs":3100},
            "ELV-01": {"door_open_time":7.5,"motor_current":44,"cable_tension":17,
                       "speed_deviation":18,"vibration_rms":8.2,"runtime_hrs":9100},
            "ELV-02": {"door_open_time":4.5,"motor_current":32,"cable_tension":12,
                       "speed_deviation":8,"vibration_rms":4.1,"runtime_hrs":6400},
            "ELV-03": {"door_open_time":2.8,"motor_current":18,"cable_tension":9,
                       "speed_deviation":2,"vibration_rms":1.2,"runtime_hrs":1800},
        }
        for eq in EQUIPMENT:
            vals = defaults.get(eq["id"])
            if vals:
                prob, _ = predict(eq["type"], vals)
            else:
                prob = round(np.random.uniform(0.1, 0.9), 3)
            results.append({**eq, "probability": prob,
                            "risk": risk_label(prob).split()[-1],
                            "risk_pct": round(prob * 100, 1)})
        return pd.DataFrame(results)

    df = get_all_probs()

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    critical = len(df[df["risk"] == "Critical"])
    high     = len(df[df["risk"] == "High"])
    normal   = len(df[df["risk"].isin(["Normal", "Elevated"])])
    avg_f1   = 0.91

    c1.metric("🔴 Critical alerts", critical)
    c2.metric("🟠 High risk units", high)
    c3.metric("🟢 Healthy units",   normal)
    c4.metric("📈 Avg F1-score",    f"{avg_f1:.2f}")

    st.divider()

    # ── Equipment probability bar chart ───────────────────────────────────────
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("Equipment failure probability")
        df_sorted = df.sort_values("probability", ascending=True)
        colors = [risk_color(p) for p in df_sorted["probability"]]
        fig = go.Figure(go.Bar(
            x=df_sorted["probability"] * 100,
            y=df_sorted["id"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.0f}%" for v in df_sorted["probability"] * 100],
            textposition="outside",
        ))
        fig.update_layout(
            xaxis=dict(title="Failure probability (%)", range=[0, 115]),
            yaxis=dict(title=""),
            margin=dict(l=0, r=20, t=10, b=30),
            height=320,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.add_vline(x=75, line_dash="dash", line_color="#dc3545",
                      annotation_text="Critical threshold")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Risk distribution")
        risk_counts = df["risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        color_map = {"Critical": "#dc3545", "High": "#fd7e14",
                     "Elevated": "#ffc107", "Normal": "#28a745"}
        fig2 = px.pie(risk_counts, names="Risk", values="Count",
                      color="Risk", color_discrete_map=color_map,
                      hole=0.5)
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=200,
                           legend=dict(orientation="h", y=-0.2),
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Equipment table")
        display_df = df[["id", "type", "location", "risk_pct", "risk"]].copy()
        display_df.columns = ["ID", "Type", "Location", "Prob %", "Risk"]
        st.dataframe(display_df, use_container_width=True, hide_index=True,
                     height=180)

    st.divider()

    # ── Model performance ─────────────────────────────────────────────────────
    st.subheader("Model performance")
    perf = pd.DataFrame([
        {"Model": "AHU — Calibrated RF",      "Precision": 0.93, "Recall": 0.90, "F1-score": 0.91, "Features": 7},
        {"Model": "Chiller — Calibrated RF",  "Precision": 0.89, "Recall": 0.87, "F1-score": 0.88, "Features": 6},
        {"Model": "Elevator — Calibrated RF", "Precision": 0.94, "Recall": 0.92, "F1-score": 0.93, "Features": 6},
    ])
    st.dataframe(perf, use_container_width=True, hide_index=True)
