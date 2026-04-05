import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from pages_code.data import EQUIPMENT, FEATURES, predict


def show():
    st.title("🌀 Scenario Simulation")
    st.caption("Stress-test equipment under degraded conditions — see failure probability over time")

    eq_options = [e["id"] for e in EQUIPMENT]
    modes = ["Gradual wear", "Sudden spike", "Seasonal stress", "Sensor drift", "Custom ramp"]

    c1, c2, c3 = st.columns(3)
    equip_id   = c1.selectbox("Equipment unit", eq_options)
    mode       = c2.selectbox("Degradation mode", modes)
    days       = c3.slider("Simulation duration (days)", 7, 180, 30)

    equip_type = next(e["type"] for e in EQUIPMENT if e["id"] == equip_id)
    feats      = FEATURES[equip_type]

    # Starting sensor values
    with st.expander("Starting sensor values (day 0)", expanded=False):
        start_vals = {}
        cols = st.columns(3)
        for i, f in enumerate(feats):
            step = 0.01 if f["max"] - f["min"] <= 2 else (0.1 if f["max"] - f["min"] <= 20 else 1.0)
            start_vals[f["name"]] = cols[i % 3].slider(
                f["label"], min_value=float(f["min"]), max_value=float(f["max"]),
                value=float(f["default"]), step=step,
                key=f"sim_sl_{equip_type}_{f['name']}"
            )

    if st.button("▶ Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Simulating degradation trajectory…"):
            rng = np.random.default_rng(42)
            probs, day_labels = [], []

            for d in range(days):
                t = d / (days - 1) if days > 1 else 0
                if mode == "Gradual wear":
                    scale = 0.15 + 0.7 * t
                elif mode == "Sudden spike":
                    scale = 0.15 if d < days * 0.65 else 0.80
                elif mode == "Seasonal stress":
                    scale = 0.2 + 0.45 * np.sin(t * np.pi)
                elif mode == "Sensor drift":
                    scale = 0.2 + 0.3 * t + 0.15 * np.sin(t * 6)
                else:  # Custom ramp
                    scale = t ** 0.7

                degraded_vals = {}
                for f in feats:
                    base = start_vals[f["name"]]
                    span = f["max"] - f["min"]
                    delta = scale * span * 0.4 * rng.uniform(0.7, 1.3)
                    # COP degrades downward, others upward
                    if f["name"] == "cop":
                        v = max(f["min"], base - delta)
                    else:
                        v = min(f["max"], base + delta)
                    degraded_vals[f["name"]] = round(float(v), 3)

                prob, _ = predict(equip_type, degraded_vals)
                probs.append(prob)
                day_labels.append(f"Day {d+1}")

        # ── Time-series chart ─────────────────────────────────────────────────
        st.subheader(f"Failure probability — {equip_id} · {mode}")
        threshold = 0.75
        colors = ["#dc3545" if p >= threshold else "#4a90d9" for p in probs]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=day_labels, y=[p * 100 for p in probs],
            mode="lines+markers",
            line=dict(color="#4a90d9", width=2),
            marker=dict(color=colors, size=6),
            fill="tozeroy", fillcolor="rgba(74,144,217,0.1)",
            name="Failure probability"
        ))
        fig.add_hline(y=75, line_dash="dash", line_color="#dc3545",
                      annotation_text="Critical threshold (75%)",
                      annotation_position="top left")
        fig.add_hline(y=50, line_dash="dot", line_color="#fd7e14",
                      annotation_text="High threshold (50%)",
                      annotation_position="top left")
        fig.update_layout(
            yaxis=dict(title="Failure probability (%)", range=[0, 105]),
            xaxis=dict(tickangle=-45, nticks=min(days, 20)),
            margin=dict(l=0, r=0, t=20, b=40),
            height=360,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Summary metrics ───────────────────────────────────────────────────
        peak_prob  = max(probs)
        peak_day   = probs.index(peak_prob) + 1
        exceed_75  = sum(1 for p in probs if p >= 0.75)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Peak probability",       f"{peak_prob*100:.1f}%")
        k2.metric("Peak on day",            peak_day)
        k3.metric("Days above critical",    exceed_75)
        k4.metric("Avg probability",        f"{np.mean(probs)*100:.1f}%")

        # ── Probability data table ────────────────────────────────────────────
        with st.expander("View simulation data"):
            sim_df = pd.DataFrame({
                "Day": range(1, days + 1),
                "Probability (%)": [round(p * 100, 1) for p in probs],
                "Risk": ["Critical" if p >= 0.75 else "High" if p >= 0.5
                         else "Elevated" if p >= 0.25 else "Normal" for p in probs]
            })
            st.dataframe(sim_df, use_container_width=True, hide_index=True)
