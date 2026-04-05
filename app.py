import sys, os

# Ensure pages_code is importable on Streamlit Cloud
ROOT = os.path.dirname(os.path.abspath(__file__))
PAGES = os.path.join(ROOT, "pages_code")
for p in [ROOT, PAGES]:
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib
import streamlit as st

st.set_page_config(
    page_title="FM AI Agent v2 — Equipment Failure Prediction",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .risk-critical { color: #dc3545; font-weight: 600; }
    .risk-high     { color: #fd7e14; font-weight: 600; }
    .risk-normal   { color: #28a745; font-weight: 600; }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🏭 FM AI Agent v2")
st.sidebar.caption("Equipment Failure Prediction")
st.sidebar.divider()

pages = {
    "📊 Dashboard":   "dashboard",
    "🔮 Prediction":  "prediction",
    "🌀 Simulation":  "simulation",
    "📡 Real-time":   "realtime",
    "🔧 Work Orders": "workorders",
    "💬 Feedback":    "feedback",
    "⚙️ AutoML":      "automl",
}

if "page" not in st.session_state:
    st.session_state.page = "dashboard"

for label, key in pages.items():
    if st.sidebar.button(label, use_container_width=True,
                         type="primary" if st.session_state.page == key else "secondary"):
        st.session_state.page = key

st.sidebar.divider()
st.sidebar.success("🟢 Agent online")
st.sidebar.caption("HVAC · Elevator · AutoML")

# Route using importlib — works reliably on Streamlit Cloud
page = st.session_state.page
mod = importlib.import_module(page)
importlib.reload(mod)
mod.show()
