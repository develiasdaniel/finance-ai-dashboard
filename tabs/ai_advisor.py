import streamlit as st

def render(user_name: str, monthly_income: float):
    st.header("🤖 AI Financial Advisor")
    st.markdown(f"Chat with your personal AI advisor, **{user_name}**. Ask anything about your finances!")

    # ─── Context Panel ───
    with st.expander("📋 Context Being Sent to AI (from other tabs)", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**From Expense Forecast:**")
            st.markdown("- Projected spend: `$—`")
            st.markdown("- Trend: `—`")
        with col2:
            st.markdown("**From Risk Score:**")
            st.markdown("- Risk Score: `—/100`")
            st.markdown("- Default Prob: `—%`")
        with col3:
            st.markdown("**From Habit Tracker:**")
            st.markdown("- Profile: `—`")
            st.markdown("- Top category: `—`")

        st.caption("🔧 This context will auto-populate once models are connected.")

    st.markdown("---")

    # ─── Chat Interface ───
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    f"👋 Hi {user_name}! I'm your AI Financial Advisor. "
                    "I can help you understand your spending patterns, risk profile, "
                    "and give you personalized recommendations.\n\n"
                    "Try asking me things like:\n"
                    '- "How much will I spend next month?"\n'
                    '- "What\'s my biggest financial risk?"\n'
                    '- "Give me a savings challenge for this month"\n'
                    '- "How can I reduce my expenses?"\n\n'
                    "🔧 *Note: I'm not connected to an LLM yet. "
                    "This is a UI skeleton.*"
                )
            }
        ]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about your finances..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Placeholder AI response
        placeholder_response = (
            "🔧 **[Placeholder Response]**\n\n"
            f'You asked: *"{prompt}"*\n\n'
            "Once connected, I will:\n"
            "1. Pull your latest expense forecast from Tab 1\n"
            "2. Check your risk score from Tab 2\n"
            "3. Review your habit profile from Tab 4\n"
            "4. Generate a personalized answer using an LLM "
            "(Gemini/GPT via LangChain)\n\n"
            "*Stay tuned!* 🚀"
        )

        st.session_state.messages.append({"role": "assistant", "content": placeholder_response})
        with st.chat_message("assistant"):
            st.markdown(placeholder_response)

    # ─── Sidebar-like quick actions ───
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Monthly Summary", key="quick_summary"):
            st.info("🔧 Will generate a monthly financial summary.")
    with col2:
        if st.button("🎯 Savings Challenge", key="quick_challenge"):
            st.info("🔧 Will generate a personalized savings challenge.")
    with col3:
        if st.button("🚨 Spending Alerts", key="quick_alerts"):
            st.info("🔧 Will show unusual spending patterns.")