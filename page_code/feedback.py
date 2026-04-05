import streamlit as st
import pandas as pd
import plotly.express as px
from data import EQUIPMENT, save_feedback, load_feedback


def show():
    st.title("💬 Feedback Loop")
    st.caption("Technician feedback · stored in SQLite · feeds model retraining pipeline")

    col_l, col_r = st.columns([1, 1])

    # ── Submit feedback form ──────────────────────────────────────────────────
    with col_l:
        st.subheader("Submit technician feedback")
        with st.form("feedback_form"):
            eq_ids = [e["id"] for e in EQUIPMENT]
            fb_equip = st.selectbox("Equipment unit", eq_ids)
            fb_outcome = st.radio(
                "Prediction outcome",
                ["True Positive", "False Positive", "False Negative", "True Negative"],
                horizontal=True,
            )
            fault_options = [
                "Bearing wear", "Coil fouling", "Refrigerant leak",
                "Belt / cable tension", "Sensor fault", "Motor overcurrent",
                "Filter clog", "Door mechanism", "No fault found", "Other"
            ]
            fb_fault = st.selectbox("Actual fault observed", fault_options)
            fb_prob  = st.slider("Model-predicted probability at time of alert (%)", 0, 100, 72)
            fb_notes = st.text_area("Technician notes", placeholder="Describe inspection findings…", height=100)
            submitted = st.form_submit_button("Submit feedback", type="primary")

        if submitted:
            save_feedback(fb_equip, fb_outcome, fb_fault, fb_notes, fb_prob / 100)
            st.success("✅ Feedback recorded in SQLite. Retraining queued.")
            st.info("In production: this triggers an automated retraining pipeline (e.g. Vertex AI / MLflow).")

    # ── Feedback history ──────────────────────────────────────────────────────
    with col_r:
        st.subheader("Feedback history")
        df = load_feedback()
        if df.empty:
            st.info("No feedback submitted yet. Use the form on the left to add the first record.")
        else:
            display = df[["ts","equipment","outcome","actual_fault","prob","notes"]].copy()
            display.columns = ["Timestamp","Equipment","Outcome","Fault","Prob","Notes"]
            display["Prob"] = display["Prob"].apply(lambda x: f"{x*100:.0f}%")
            st.dataframe(display, use_container_width=True, hide_index=True, height=320)

    st.divider()

    # ── Analytics on feedback ─────────────────────────────────────────────────
    st.subheader("Feedback analytics")
    df = load_feedback()
    if len(df) < 2:
        # Show demo data when DB is empty
        demo = [
            {"outcome":"True Positive","actual_fault":"Bearing wear","equipment":"AHU-01","prob":0.83},
            {"outcome":"False Positive","actual_fault":"No fault found","equipment":"CHR-02","prob":0.61},
            {"outcome":"True Positive","actual_fault":"Coil fouling","equipment":"AHU-02","prob":0.72},
            {"outcome":"True Positive","actual_fault":"Belt / cable tension","equipment":"ELV-01","prob":0.91},
            {"outcome":"False Negative","actual_fault":"Refrigerant leak","equipment":"CHR-01","prob":0.38},
            {"outcome":"True Positive","actual_fault":"Filter clog","equipment":"AHU-01","prob":0.77},
            {"outcome":"True Negative","actual_fault":"No fault found","equipment":"AHU-03","prob":0.18},
        ]
        df = pd.DataFrame(demo)
        st.caption("Showing demo data — submit feedback to see your own analytics.")

    c1, c2 = st.columns(2)
    with c1:
        outcome_counts = df["outcome"].value_counts().reset_index()
        outcome_counts.columns = ["Outcome", "Count"]
        color_map = {
            "True Positive": "#28a745", "False Positive": "#fd7e14",
            "False Negative": "#dc3545", "True Negative": "#4a90d9"
        }
        fig = px.bar(outcome_counts, x="Outcome", y="Count",
                     color="Outcome", color_discrete_map=color_map,
                     title="Prediction outcomes")
        fig.update_layout(showlegend=False, margin=dict(l=0,r=0,t=40,b=20),
                          height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fault_counts = df["actual_fault"].value_counts().reset_index()
        fault_counts.columns = ["Fault","Count"]
        fig2 = px.pie(fault_counts, names="Fault", values="Count",
                      title="Fault type distribution", hole=0.4)
        fig2.update_layout(margin=dict(l=0,r=0,t=40,b=10), height=280,
                            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    # Retraining readiness
    st.divider()
    st.subheader("Retraining readiness")
    total   = len(df)
    tp      = len(df[df["outcome"] == "True Positive"]) if "outcome" in df else 0
    accuracy = tp / total if total > 0 else 0
    r1, r2, r3 = st.columns(3)
    r1.metric("Total feedback records",  total)
    r2.metric("True positive rate",      f"{accuracy*100:.0f}%")
    r3.metric("Retraining threshold",    "50 records")
    if total >= 50:
        st.success("✅ Enough feedback collected — retraining pipeline can be triggered.")
    else:
        st.progress(min(total / 50, 1.0), text=f"{total}/50 records needed for retraining")
