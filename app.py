import streamlit as st

st.set_page_config(
    page_title="FM AI Agent v2 — Equipment Failure Prediction",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
    }
    .risk-critical { color: #dc3545; font-weight: 600; }
    .risk-high     { color: #fd7e14; font-weight: 600; }
    .risk-normal   { color: #28a745; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 4px 4px 0 0; }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🏭 FM AI Agent v2")
st.sidebar.caption("Equipment Failure Prediction")
st.sidebar.divider()

pages = {
    "📊 Dashboard":    "dashboard",
    "🔮 Prediction":   "prediction",
    "🌀 Simulation":   "simulation",
    "📡 Real-time":    "realtime",
    "🔧 Work Orders":  "workorders",
    "💬 Feedback":     "feedback",
    "⚙️ AutoML":       "automl",
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

# Route to pages
page = st.session_state.page

if page == "dashboard":
    from pages_code import dashboard; dashboard.show()
elif page == "prediction":
    from pages_code import prediction; prediction.show()
elif page == "simulation":
    from pages_code import simulation; simulation.show()
elif page == "realtime":
    from pages_code import realtime; realtime.show()
elif page == "workorders":
    from pages_code import workorders; workorders.show()
elif page == "feedback":
    from pages_code import feedback; feedback.show()
elif page == "automl":
    from pages_code import automl; automl.show()
