import streamlit as st

def render(user_name: str, monthly_income: float):
    st.header("🛡️ Risk Score")
    st.markdown(f"**{user_name}**, let's evaluate your financial risk profile.")

    # ─── Input Section ───
    st.markdown("### 📝 Your Financial Information")

    col1, col2 = st.columns(2)
    with col1:
        loan_amount = st.number_input("Total Loan Amount ($)", value=10000, step=500, key="loan_amt")
        credit_history = st.selectbox("Credit History", ["Good", "Bad"], key="credit_hist")
        employment = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"], key="employment")
    with col2:
        monthly_debt = st.number_input("Monthly Debt Payment ($)", value=500, step=50, key="monthly_debt")
        dependents = st.number_input("Number of Dependents", value=0, step=1, key="dependents")
        savings = st.number_input("Monthly Savings ($)", value=500, step=50, key="savings")

    # ─── Calculated Ratios (preview) ───
    st.markdown("### 📐 Calculated Ratios")
    col1, col2, col3 = st.columns(3)
    dti = round(monthly_debt / monthly_income, 2) if monthly_income > 0 else 0
    savings_rate = round(savings / monthly_income, 2) if monthly_income > 0 else 0

    with col1:
        st.metric("Debt-to-Income Ratio", f"{dti:.0%}")
    with col2:
        st.metric("Savings Rate", f"{savings_rate:.0%}")
    with col3:
        st.metric("EMI Burden", "—", delta="Coming soon")

    st.markdown("---")

    # ─── Model Selection ───
    st.markdown("### 🧠 Model Selection")
    risk_model = st.radio(
        "Choose classification model:",
        ["Logistic Regression (Baseline)", "XGBoost (Advanced)"],
        key="risk_model"
    )

    # ─── Placeholder: Risk Score Output ───
    st.markdown("### 🎯 Your Risk Score")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### 🚦 Risk Gauge")
        st.markdown(
            """
            <div style="text-align:center; padding:40px; background:#1e1e2f; border-radius:16px;">
                <h1 style="font-size:72px; color:#555;">—</h1>
                <p style="color:#888;">Risk Score (0–100)</p>
                <p style="color:#666;">🔧 Connect model to calculate</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown("#### 📊 Default Probability")
        st.markdown(
            """
            <div style="text-align:center; padding:40px; background:#1e1e2f; border-radius:16px;">
                <h1 style="font-size:72px; color:#555;">—%</h1>
                <p style="color:#888;">Probability of Default</p>
                <p style="color:#666;">🔧 Connect model to calculate</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ─── Placeholder: SHAP Explanation ───
    with st.expander("🔍 SHAP Analysis — Top Factors Affecting Your Score"):
        st.markdown("""
        **Top 3 contributing factors will appear here:**

        1. 🔴 `debt_to_income_ratio` — **??%** impact
        2. 🟡 `credit_history` — **??%** impact
        3. 🟢 `income` — **??%** impact

        🔧 *SHAP chart will render here after model integration.*
        """)

    # ─── Placeholder: Model Metrics ───
    with st.expander("📋 Model Evaluation Metrics"):
        st.markdown("""
        | Metric    | Logistic Regression | XGBoost |
        |-----------|---------------------|---------|
        | F1-Score  | —                   | —       |
        | ROC-AUC   | —                   | —       |
        | Accuracy  | —                   | —       |
        """)