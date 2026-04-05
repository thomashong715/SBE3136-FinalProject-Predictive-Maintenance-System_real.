import streamlit as st
import pandas as pd
import datetime


WORK_ORDERS = [
    {"id": "WO-2041", "equipment": "AHU-01",  "type": "Corrective",
     "description": "Bearing replacement — fan vibration exceeded 10 mm/s",
     "priority": "Critical", "status": "Open",
     "risk_score": 0.82, "est_cost": 1200,
     "created": "2025-04-04", "due": "2025-04-05"},
    {"id": "WO-2042", "equipment": "ELV-01",  "type": "Safety",
     "description": "Door sensor recalibration — dwell time 7.5 s (limit 6 s)",
     "priority": "Critical", "status": "In Progress",
     "risk_score": 0.91, "est_cost": 800,
     "created": "2025-04-04", "due": "2025-04-05"},
    {"id": "WO-2043", "equipment": "CHR-01",  "type": "Corrective",
     "description": "Refrigerant pressure check — 20 bar anomaly detected",
     "priority": "High", "status": "Open",
     "risk_score": 0.74, "est_cost": 2500,
     "created": "2025-04-03", "due": "2025-04-06"},
    {"id": "WO-2044", "equipment": "AHU-02",  "type": "Preventive",
     "description": "Filter replacement — ΔP at 220 Pa, trending upward",
     "priority": "High", "status": "Open",
     "risk_score": 0.58, "est_cost": 350,
     "created": "2025-04-03", "due": "2025-04-07"},
    {"id": "WO-2045", "equipment": "ELV-02",  "type": "Preventive",
     "description": "Cable tension inspection — early wear signal detected",
     "priority": "High", "status": "Scheduled",
     "risk_score": 0.47, "est_cost": 600,
     "created": "2025-04-02", "due": "2025-04-08"},
    {"id": "WO-2046", "equipment": "AHU-03",  "type": "Routine",
     "description": "Quarterly coil cleaning and airflow validation",
     "priority": "Medium", "status": "Scheduled",
     "risk_score": 0.21, "est_cost": 450,
     "created": "2025-04-01", "due": "2025-04-12"},
]


def priority_color(p):
    return {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(p, "⚪")


def show():
    st.title("🔧 Work Orders")
    st.caption("AI-generated maintenance tasks · prioritized by risk score × estimated cost")

    df = pd.DataFrame(WORK_ORDERS)

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total open orders",      len(df[df["status"].isin(["Open","In Progress"])]))
    k2.metric("🔴 Critical",             len(df[df["priority"] == "Critical"]))
    k3.metric("Est. total cost",        f"${df['est_cost'].sum():,}")
    k4.metric("Est. cost savings (5×)", f"${df['est_cost'].sum() * 5:,}")

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)
    sel_priority = fc1.multiselect("Priority", ["Critical", "High", "Medium"], default=["Critical","High","Medium"])
    sel_status   = fc2.multiselect("Status", ["Open","In Progress","Scheduled","Closed"], default=["Open","In Progress","Scheduled"])
    sel_type     = fc3.multiselect("Type", ["Safety","Corrective","Preventive","Routine"], default=["Safety","Corrective","Preventive","Routine"])

    mask = (
        df["priority"].isin(sel_priority) &
        df["status"].isin(sel_status) &
        df["type"].isin(sel_type)
    )
    filtered = df[mask].sort_values(["priority", "risk_score"], ascending=[True, False])

    # ── Work order cards ──────────────────────────────────────────────────────
    priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    filtered = filtered.copy()
    filtered["_pord"] = filtered["priority"].map(priority_order)
    filtered = filtered.sort_values(["_pord", "risk_score"], ascending=[True, False])

    for _, row in filtered.iterrows():
        ico = priority_color(row["priority"])
        with st.expander(f"{ico} **{row['id']}** · {row['equipment']} · {row['priority']} — {row['description'][:60]}…"):
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**ID:** `{row['id']}`")
            c2.markdown(f"**Equipment:** {row['equipment']}")
            c3.markdown(f"**Type:** {row['type']}")
            c4.markdown(f"**Status:** {row['status']}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Risk score",  f"{row['risk_score']*100:.0f}%")
            c6.metric("Est. cost",   f"${row['est_cost']:,}")
            c7.markdown(f"**Created:** {row['created']}")
            c8.markdown(f"**Due:** {row['due']}")

            st.markdown(f"**Full description:** {row['description']}")

            act1, act2, act3 = st.columns(3)
            if act1.button("✅ Mark complete", key=f"done_{row['id']}"):
                st.success(f"{row['id']} marked as complete.")
            if act2.button("📋 Assign technician", key=f"assign_{row['id']}"):
                st.info("Technician assignment UI — connect to HR/BMS system.")
            if act3.button("🗑 Close order", key=f"close_{row['id']}"):
                st.warning(f"{row['id']} closed.")

    st.divider()

    # ── Generate new WO from prediction ──────────────────────────────────────
    st.subheader("➕ Generate work order from prediction")
    with st.form("new_wo_form"):
        wc1, wc2 = st.columns(2)
        wo_equip = wc1.selectbox("Equipment", ["AHU-01","AHU-02","AHU-03","CHR-01","CHR-02","ELV-01","ELV-02","ELV-03"])
        wo_type  = wc2.selectbox("Work type", ["Corrective","Preventive","Safety","Routine"])
        wo_desc  = st.text_area("Description / findings", placeholder="Describe the fault or maintenance task…")
        wo_cost  = st.number_input("Estimated cost ($)", min_value=0, value=500, step=50)
        wo_due   = st.date_input("Due date", value=datetime.date.today() + datetime.timedelta(days=3))
        submitted = st.form_submit_button("Create work order", type="primary")
        if submitted and wo_desc:
            new_id = f"WO-{2047 + len(WORK_ORDERS)}"
            st.success(f"✅ Work order **{new_id}** created for {wo_equip} — due {wo_due}. (In production this writes to CMMS/BMS.)")
