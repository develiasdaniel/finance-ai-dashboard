import streamlit as st

def render(user_name: str, monthly_income: float):
    st.header("📊 Habit Tracker")
    st.markdown(f"Track your financial habits and build better ones, **{user_name}**!")

    # ─── Profile Section ───
    st.markdown("### 🏷️ Your Spending Profile")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            """
            <div style="text-align:center; padding:30px; background:#1e1e2f;
                        border-radius:16px; border: 2px solid #444;">
                <h2 style="color:#f0f0f0;">🔧 TBD</h2>
                <p style="color:#aaa; font-size:14px;">Your Profile Label</p>
                <p style="color:#666; font-size:12px;">
                    (e.g., Impulse Spender, Consistent Saver, High Fixed Cost)
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown("**How is your profile determined?**")
        st.markdown("""
        Your spending profile is calculated using rule-based classification on:
        - 📦 Average spend per category
        - 🔄 Transaction frequency per category
        - 📉 Monthly spending variability

        🔧 *Profile engine coming soon!*
        """)

    st.markdown("---")

    # ─── Habit Goals ───
    st.markdown("### 🎯 Your Financial Goals")

    # Goal input
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        goal_category = st.selectbox(
            "Category",
            ["Food & Delivery", "Entertainment", "Shopping", "Transport", "Subscriptions", "Other"],
            key="goal_cat"
        )
    with col2:
        goal_limit = st.number_input("Monthly Limit ($)", value=100, step=10, key="goal_limit")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        add_goal = st.button("➕ Add Goal", key="add_goal")

    # Initialize goals in session state
    if "goals" not in st.session_state:
        st.session_state.goals = [
            {"category": "Food & Delivery", "limit": 150, "spent": 0},
            {"category": "Entertainment", "limit": 80, "spent": 0},
        ]

    if add_goal:
        st.session_state.goals.append({
            "category": goal_category,
            "limit": goal_limit,
            "spent": 0
        })
        st.success(f"✅ Goal added: {goal_category} < ${goal_limit}/month")

    # Display goals with progress bars
    st.markdown("### 📊 Goal Progress")
    for i, goal in enumerate(st.session_state.goals):
        col1, col2 = st.columns([3, 1])
        with col1:
            progress = goal["spent"] / goal["limit"] if goal["limit"] > 0 else 0
            st.markdown(f"**{goal['category']}** — ${goal['spent']} / ${goal['limit']}")
            st.progress(min(progress, 1.0))
        with col2:
            remaining = goal["limit"] - goal["spent"]
            if remaining > 0:
                st.markdown(f"✅ ${remaining} left")
            else:
                st.markdown("🔴 Over budget!")

    st.markdown("---")

    # ─── Streaks & Achievements ───
    st.markdown("### 🔥 Streaks & Achievements")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div style="text-align:center; padding:20px; background:#1e1e2f; border-radius:12px;">
                <h2>🔥 0</h2>
                <p style="color:#aaa; font-size:12px;">Day Streak</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div style="text-align:center; padding:20px; background:#1e1e2f; border-radius:12px;">
                <h2>🏆 0</h2>
                <p style="color:#aaa; font-size:12px;">Goals Met</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div style="text-align:center; padding:20px; background:#1e1e2f; border-radius:12px;">
                <h2>💰 $0</h2>
                <p style="color:#aaa; font-size:12px;">Saved This Month</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <div style="text-align:center; padding:20px; background:#1e1e2f; border-radius:12px;">
                <h2>📈 —</h2>
                <p style="color:#aaa; font-size:12px;">Best Category</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ─── Weekly Overview Placeholder ───
    with st.expander("📅 Weekly Spending Overview"):
        st.markdown("""
        | Day       | Food | Transport | Entertainment | Shopping |
        |-----------|------|-----------|---------------|----------|
        | Monday    | —    | —         | —             | —        |
        | Tuesday   | —    | —         | —             | —        |
        | Wednesday | —    | —         | —             | —        |
        | Thursday  | —    | —         | —             | —        |
        | Friday    | —    | —         | —             | —        |
        | Saturday  | —    | —         | —             | —        |
        | Sunday    | —    | —         | —             | —        |

        🔧 *Will populate when connected to transaction data.*
        """)