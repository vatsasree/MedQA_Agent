import streamlit as st
import os
import sys

# Ensure src module can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent import build_agent
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="MedQA Assistant", page_icon="⚕️", layout="wide")

st.title("⚕️ Medical Clinical Insight Engine (MedQA)")
st.write("Ask questions about clinical guidelines (from the medical handbook) or specific patient histories (from patient reports).")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the agent
if "agent" not in st.session_state:
    with st.spinner("Initializing MedQA Agent..."):
        try:
            # We allow user to specify LLM via code, default is open-source
            st.session_state.agent = build_agent(model_type="open-source", model_name="Qwen/Qwen3.5-122B-A10B")
            st.toast("Agent Initialized successfully!", icon="✅")
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            st.session_state.agent = None

# Sidebar configurations
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Model settings are loaded from `.env`.")
    st.info("Ensure you have run `python src/ingest.py` to ingest the PDFs before querying.")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("E.g., What are the standard treatments for hypertension?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare input for agent
    inputs = {"messages": [("user", prompt)]}
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking and searching medical records..."):
            try:
                if st.session_state.agent:
                    # Run the agent
                    response = st.session_state.agent.invoke(inputs)
                    
                    # The final response is the last message in the 'messages' list
                    final_reply = response["messages"][-1].content
                    st.markdown(final_reply)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": final_reply})
                else:
                    st.error("Agent is not initialized. Please check the logs.")
            except Exception as e:
                st.error(f"Error during agent execution: {e}")
