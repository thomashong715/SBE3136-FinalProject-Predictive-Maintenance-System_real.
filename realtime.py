import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from data import EQUIPMENT, FEATURES, predict, risk_color


def show():
    st.title("📡 Real-time Sensor Feed")
    st.caption("Live time-series dataset replay — simulating sensor monitoring")

    eq_options = [e["id"] for e in EQUIPMENT if e["type"] == "AHU"]
    equip_id   = st.selectbox("Select equipment unit", eq_options)
    equip_type = "AHU"
    feats      = FEATURES[equip_type]

    # Base sensor values
    BASE = {
        "AHU-01": {"supply_air_temp": 28, "return_air_temp": 32, "filter_dp": 210,
                   "fan_vibration": 7, "coil_fouling": 0.6, "runtime_hrs": 6800, "ambient_humidity": 62},
        "AHU-02": {"supply_air_temp": 24, "return_air_temp": 28, "filter_dp": 150,
                   "fan_vibration": 4, "coil_fouling": 0.3, "runtime_hrs": 4500, "ambient_humidity": 55},
        "AHU-03": {"supply_air_temp": 21, "return_air_temp": 25, "filter_dp": 110,
                   "fan_vibration": 2, "coil_fouling": 0.15, "runtime_hrs": 2000, "ambient_humidity": 50},
    }
    base = BASE.get(equip_id, BASE["AHU-01"])

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    n_ticks    = col_ctrl1.slider("Ticks to display", 20, 100, 50)
    interval   = col_ctrl2.selectbox("Update interval", ["0.3s", "0.6s", "1.0s"], index=1)
    noise_lvl  = col_ctrl3.selectbox("Noise level", ["Low", "Medium", "High"], index=1)

    noise_map  = {"Low": 0.02, "Medium": 0.06, "High": 0.14}
    noise      = noise_map[noise_lvl]
    sleep_map  = {"0.3s": 0.3, "0.6s": 0.6, "1.0s": 1.0}
    sleep_secs = sleep_map[interval]

    # Session-state history
    if "rt_history" not in st.session_state or st.session_state.get("rt_equip") != equip_id:
        st.session_state.rt_history = {f["name"]: [] for f in feats}
        st.session_state.rt_history["prob"] = []
        st.session_state.rt_tick = 0
        st.session_state.rt_equip = equip_id

    start = st.button("▶ Start streaming", type="primary")
    stop  = st.button("⏹ Stop")

    # ── Live placeholders ─────────────────────────────────────────────────────
    sensor_ph = st.empty()
    chart_ph  = st.empty()
    prob_ph   = st.empty()
    log_ph    = st.empty()

    if start and not stop:
        rng = np.random.default_rng()
        for _ in range(300):  # max iterations
            vals = {}
            for f in feats:
                b = base[f["name"]]
                span = f["max"] - f["min"]
                v = b + rng.uniform(-noise * span, noise * span)
                v = float(np.clip(v, f["min"], f["max"]))
                vals[f["name"]] = round(v, 2)
                st.session_state.rt_history[f["name"]].append(v)

            prob, _ = predict(equip_type, vals)
            st.session_state.rt_history["prob"].append(prob)
            st.session_state.rt_tick += 1

            # Keep only last n_ticks
            for k in st.session_state.rt_history:
                if len(st.session_state.rt_history[k]) > n_ticks:
                    st.session_state.rt_history[k] = st.session_state.rt_history[k][-n_ticks:]

            # Sensor cards
            with sensor_ph.container():
                st.subheader(f"Live sensor readings — {equip_id}  🟢")
                scols = st.columns(4)
                sensor_display = [
                    ("Supply air temp", vals["supply_air_temp"], "°C"),
                    ("Return air temp", vals["return_air_temp"], "°C"),
                    ("Filter ΔP", vals["filter_dp"], "Pa"),
                    ("Fan vibration",  vals["fan_vibration"],  "mm/s"),
                    ("Coil fouling",   vals["coil_fouling"],   "idx"),
                    ("Humidity",       vals["ambient_humidity"],"%"),
                    ("Runtime",        vals["runtime_hrs"],    "hrs"),
                    (f"Tick",          st.session_state.rt_tick, ""),
                ]
                for i, (name, val, unit) in enumerate(sensor_display):
                    scols[i % 4].metric(f"{name}", f"{val:.2f} {unit}")

            # Probability time-series chart
            with chart_ph.container():
                probs_hist = st.session_state.rt_history["prob"]
                col_line   = risk_color(prob)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=[p * 100 for p in probs_hist],
                    mode="lines", line=dict(color=col_line, width=2),
                    fill="tozeroy", fillcolor="rgba(74,144,217,0.1)",
                    name="Failure probability"
                ))
                fig.add_hline(y=75, line_dash="dash", line_color="#dc3545")
                fig.add_hline(y=50, line_dash="dot",  line_color="#fd7e14")
                fig.update_layout(
                    yaxis=dict(title="Failure probability (%)", range=[0, 105]),
                    xaxis=dict(title="Recent ticks"),
                    margin=dict(l=0, r=0, t=10, b=30), height=220,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Prob KPI
            with prob_ph.container():
                pp1, pp2 = st.columns(2)
                pp1.metric("Current failure probability", f"{prob*100:.1f}%")
                pp2.metric("Risk", ("🔴 Critical" if prob >= 0.75 else
                                    "🟠 High" if prob >= 0.5 else
                                    "🟡 Elevated" if prob >= 0.25 else "🟢 Normal"))

            # Event log
            with log_ph.container():
                events = []
                if vals["filter_dp"] > 280:
                    events.append(f"⚠️  Filter ΔP = {vals['filter_dp']:.0f} Pa — threshold exceeded")
                if vals["fan_vibration"] > 9:
                    events.append(f"⚠️  Fan vibration = {vals['fan_vibration']:.1f} mm/s — elevated")
                if prob >= 0.75:
                    events.append(f"🔴  Failure probability {prob*100:.0f}% — critical alert")
                if events:
                    st.warning("\n".join(events))

            time.sleep(sleep_secs)
            if stop:
                break
