import os
import openai
from dotenv import load_dotenv
import streamlit as st
import datetime
import uuid
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError('OPENAI_API_KEY not found in environment variables.')

openai.api_key = api_key

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")  # Add your MongoDB URI to your .env file
client = MongoClient(MONGO_URI)
db = client["chat_rag_db"]

def chat_with_openai(prompt, model="gpt-3.5-turbo", system_message="You are a helpful AI assistant."):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Chat history functions
def save_message(role, message, session_id="default_session"):
    db.chat_history.insert_one({
        "session_id": session_id,
        "role": role,
        "message": message,
        "timestamp": datetime.datetime.utcnow()
    })

def get_history(session_id="default_session"):
    return [
        (doc["role"], doc["message"]) 
        for doc in db.chat_history.find({"session_id": session_id}).sort("timestamp", 1)
    ]

def main():
    st.set_page_config(page_title="AYAI Chatbot", page_icon="🤖")
    st.title("AYAI Chatbot")
    st.markdown("""
    <style>
    .big-font {
        font-size:28px !important;
        font-weight: bold;
        color: #4F8BF9;
    }
    </style>
    <div class='big-font'>Chat with OpenAI's GPT model in real time!</div>
    """, unsafe_allow_html=True)
    st.sidebar.header("Chatbot Persona")
    persona = st.sidebar.text_area(
        "System Message (Persona)",
        value="You are a helpful AI assistant.",
        help="Define the personality or behavior of the chatbot."
    )
    # Remove the hidden input workaround and use on_change for the main input
    def on_enter():
        st.session_state["trigger_send"] = True

    user_input = st.text_input("You:", "", key="user_input", on_change=on_enter)
    send = st.button("Send") or st.session_state.get("trigger_send", False)
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    session_id = st.session_state["session_id"]
    if user_input and send:
        with st.spinner("Waiting for response..."):
            reply = chat_with_openai(user_input, system_message=persona)
        save_message("user", user_input, session_id=session_id)
        save_message("assistant", reply, session_id=session_id)
        st.session_state["trigger_send"] = False
        st.session_state["user_input"] = ""  # Clear input after sending

    # Display chat history
    st.markdown("### Chat History")
    history = get_history(session_id=session_id)
    st.write("DEBUG: chat history list:", history)  # Debug output
    for role, message in history:
        if role == "user":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**OpenAI:** {message}")

if __name__ == "__main__":
    main()
