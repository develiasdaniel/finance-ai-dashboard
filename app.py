import streamlit as st

# ─── Page Config ───
st.set_page_config(
    page_title="💰 AI Finance Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS (opcional) ───
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ─── Sidebar ───
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/money-bag.png", width=80)
    st.title("AI Finance Dashboard")
    st.markdown("---")
    st.markdown("### 👤 User Profile")
    user_name = st.text_input("Your Name", value="Daniel")
    monthly_income = st.number_input("Monthly Income ($)", value=5000, step=100)
    st.markdown("---")
    st.caption("v0.1.0 — Cascarón Inicial")
    st.caption("Built with ❤️ and Streamlit")

# ─── Main Tabs ───
tab1, tab2, tab3, tab4 = st.tabs([
    "🗂️ Expense Forecast",
    "🛡️ Risk Score",
    "🤖 AI Financial Advisor",
    "📊 Habit Tracker"
])

# ─── Import and render each tab ───
from tabs.expense_forecast import render as render_expense_forecast
from tabs.risk_score import render as render_risk_score
from tabs.ai_advisor import render as render_ai_advisor
from tabs.habit_tracker import render as render_habit_tracker

with tab1:
    render_expense_forecast(user_name, monthly_income)

with tab2:
    render_risk_score(user_name, monthly_income)

with tab3:
    render_ai_advisor(user_name, monthly_income)

with tab4:
    render_habit_tracker(user_name, monthly_income)