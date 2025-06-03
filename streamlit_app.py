import streamlit as st
import requests

st.set_page_config(page_title="RAG Financial QA", layout="centered")

# 🎨 Branding
st.caption("Icon by Zulfa Mahendra via Flaticon")
st.image("business-and-finance.png", width=100)
st.title("💬 RAG-Powered Financial Q&A")
st.markdown("_Ask any question related to FX risk, pricing, servicing, or market reports._")

# 💬 User input
question = st.text_input("🔍 Enter your question:")

if st.button("Submit"):
    if question.strip():
        with st.spinner("Thinking..."):
            try:
                response = requests.post("http://localhost:8000/ask", json={"question": question})
                response.raise_for_status()  # Raise exception for HTTP errors
                data = response.json()

                if "answer" in data:
                    st.subheader("🧠 Answer")
                    st.success(data["answer"])
                    st.subheader(f"📄 Context ({data.get('context_count', 0)} chunks)")
                    context = data.get("context", [])
                    if context:
                        for i, chunk in enumerate(context, start=1):
                            st.markdown(f"**{i}.** {chunk}")
                    else:
                        st.write("No context chunks available.")
                else:
                    st.error(data.get("error", "Something went wrong."))
            except requests.RequestException as e:
                st.error(f"Failed to connect to backend: {str(e)}")

# Load previous queries
if "query_history" not in st.session_state:
    try:
        logs = requests.get("http://localhost:8000/logs").json()
        st.session_state.query_history = [row["question"] for row in logs[::-1]]
    except:
        st.session_state.query_history = []

# Dropdown
if st.session_state.query_history:
    selected = st.selectbox("📜 Pick a previous question:", st.session_state.query_history)
    if st.button("Re-run Selected"):
        question = selected
        with st.spinner("Thinking..."):
            try:
                response = requests.post("http://localhost:8000/ask", json={"question": question})
                response.raise_for_status()
                data = response.json()

                if "answer" in data:
                    st.subheader("🧠 Answer")
                    st.success(data["answer"])
                    st.subheader(f"📄 Context ({data.get('context_count', 0)} chunks)")
                    context = data.get("context", [])
                    if context:
                        for i, chunk in enumerate(context, start=1):
                            st.markdown(f"**{i}.** {chunk}")
                    else:
                        st.write("No context chunks available.")
                else:
                    st.error(data.get("error", "Something went wrong."))
            except requests.RequestException as e:
                st.error(f"Failed to connect to backend: {str(e)}")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | RAG Financial QA")