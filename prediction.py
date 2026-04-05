import streamlit as st
import plotly.graph_objects as go
from pages_code.data import EQUIPMENT, FEATURES, predict, risk_label, risk_color, diagnose


def show():
    st.title("🔮 Failure Prediction")
    st.caption("Enter sensor readings — AI agent returns failure probability, SHAP explainability & diagnosis")

    # ── Equipment selector ────────────────────────────────────────────────────
    eq_types = ["AHU", "Chiller", "Elevator"]
    eq_ids   = {t: [e["id"] for e in EQUIPMENT if e["type"] == t] for t in eq_types}

    col_t, col_e = st.columns(2)
    with col_t:
        equip_type = st.selectbox("Equipment type", eq_types)
    with col_e:
        equip_id = st.selectbox("Equipment unit", eq_ids[equip_type])

    st.divider()

    # ── Sensor sliders ────────────────────────────────────────────────────────
    feats = FEATURES[equip_type]
    n_cols = 2
    cols = st.columns(n_cols)
    values = {}
    for i, f in enumerate(feats):
        step = 0.01 if f["max"] - f["min"] <= 2 else (0.1 if f["max"] - f["min"] <= 20 else 1.0)
        values[f["name"]] = cols[i % n_cols].slider(
            f["label"], min_value=float(f["min"]), max_value=float(f["max"]),
            value=float(f["default"]), step=step, key=f"sl_{equip_type}_{f['name']}"
        )

    st.divider()

    # ── Run prediction ────────────────────────────────────────────────────────
    if st.button("▶ Run Prediction", type="primary", use_container_width=True):
        with st.spinner("Running ML model + rule-based agent…"):
            prob, shap_vals = predict(equip_type, values)
            rules = diagnose(equip_type, values)

        pct   = round(prob * 100, 1)
        color = risk_color(prob)
        risk  = risk_label(prob)

        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("Failure probability", f"{pct}%")
        k2.metric("Risk classification", risk)
        k3.metric("Confidence", "High" if abs(prob - 0.5) > 0.2 else "Medium")

        col_l, col_r = st.columns(2)

        # SHAP bar chart
        with col_l:
            st.subheader("SHAP feature importance")
            sorted_shap = sorted(shap_vals.items(), key=lambda x: x[1], reverse=True)
            feat_names  = [k for k, _ in sorted_shap]
            feat_labels = [next(f["label"] for f in feats if f["name"] == k) for k in feat_names]
            feat_vals   = [v for _, v in sorted_shap]
            bar_colors  = [color if i == 0 else ("#fd7e14" if i < 3 else "#4a90d9")
                           for i in range(len(feat_vals))]
            fig = go.Figure(go.Bar(
                x=feat_vals, y=feat_labels, orientation="h",
                marker_color=bar_colors,
                text=[f"+{v:.3f}" for v in feat_vals], textposition="outside"
            ))
            fig.update_layout(
                xaxis_title="SHAP contribution",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=0, r=20, t=10, b=20),
                height=280,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Agent decision panel
        with col_r:
            st.subheader("AI agent decision")

            # Probability gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix": "%", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 25],  "color": "#d4edda"},
                        {"range": [25, 50], "color": "#fff3cd"},
                        {"range": [50, 75], "color": "#ffe5d0"},
                        {"range": [75, 100],"color": "#f8d7da"},
                    ],
                    "threshold": {"line": {"color": "#dc3545", "width": 3}, "value": 75}
                }
            ))
            fig_g.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=10),
                                 paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_g, use_container_width=True)

            st.markdown("**Rule-based diagnosis**")
            for r in rules:
                st.write(r)

            st.markdown("**Recommended action**")
            if prob >= 0.75:
                st.error(f"🚨 Immediate maintenance required for **{equip_id}**. Generate work order now.")
            elif prob >= 0.50:
                st.warning(f"⚠️ Schedule inspection within 48 hours for **{equip_id}**.")
            elif prob >= 0.25:
                st.info(f"📋 Monitor closely. Review again in 72 hours for **{equip_id}**.")
            else:
                st.success(f"✅ **{equip_id}** is operating within safe limits.")
