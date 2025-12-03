import os
from dotenv import load_dotenv
import streamlit as st
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not found. Set it in .env or environment variables.")

client = Groq(api_key=api_key)
MODEL_NAME = "llama-3.3-70b-versatile"

st.set_page_config(page_title="Groq LLM Chatbot", page_icon="🤖")

st.title("🤖 Simple Groq LLM Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant.",
        }
    ]

# Show chat history (only user + assistant)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# Input box
user_prompt = st.chat_input("Ask something...")

if user_prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=st.session_state.messages,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            reply = f"Error calling Groq API: {e}"

        st.markdown(reply)
        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": reply})
