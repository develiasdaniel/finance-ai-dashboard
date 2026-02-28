import streamlit as st
import pandas as pd
import numpy as np

def render(user_name: str, monthly_income: float):
    st.header("🗂️ Expense Forecast")
    st.markdown(f"Welcome **{user_name}**! Here you'll see your predicted expenses for the next 1–3 months.")

    # ─── Placeholder: Data Upload Section ───
    st.markdown("### 📤 Upload Your Transaction Data")
    uploaded_file = st.file_uploader(
        "Upload CSV (columns: date, category, amount)",
        type=["csv"],
        key="expense_upload"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} transactions!")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("⬆️ Upload your data to get started, or use demo data below.")

        # ─── Demo Data for skeleton ───
        if st.button("🎲 Load Demo Data", key="demo_expense"):
            np.random.seed(42)
            months = pd.date_range("2025-01-01", periods=12, freq="MS")
            demo_data = pd.DataFrame({
                "date": np.repeat(months, 5),
                "category": np.tile(["Food", "Transport", "Entertainment", "Utilities", "Shopping"], 12),
                "amount": np.random.uniform(50, 500, 60).round(2)
            })
            st.session_state["expense_demo"] = demo_data

    # ─── Show demo data if loaded ───
    if "expense_demo" in st.session_state:
        df = st.session_state["expense_demo"]
        st.dataframe(df.head(15), use_container_width=True)

    st.markdown("---")

    # ─── Placeholder: Model Selection ───
    st.markdown("### 🧠 Model Selection")
    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.radio(
            "Choose forecasting model:",
            ["ARIMA (Baseline)", "LSTM/GRU (Advanced)"],
            key="forecast_model"
        )
    with col2:
        forecast_horizon = st.slider("Forecast Horizon (months)", 1, 3, 1, key="forecast_horizon")

    # ─── Placeholder: Results Section ───
    st.markdown("### 📈 Forecast Results")
    st.markdown("---")

    # Placeholder metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Projected Next Month", value="$—", delta="Coming soon")
    with col2:
        st.metric(label="MAE", value="—", delta="Coming soon")
    with col3:
        st.metric(label="RMSE", value="—", delta="Coming soon")

    # Placeholder chart area
    st.markdown("#### 📊 Historical vs Projected Spending")
    st.empty()  # Placeholder for the actual chart
    st.caption("🔧 Chart will render here once the model is connected.")

    # ─── Placeholder: Model Comparison ───
    with st.expander("📋 Model Comparison (ARIMA vs LSTM)"):
        st.markdown("""
        | Metric | ARIMA | LSTM |
        |--------|-------|------|
        | MAE    | —     | —    |
        | RMSE   | —     | —    |
        | Training Time | — | — |
        """)
        st.caption("🔧 Will populate after model training.")