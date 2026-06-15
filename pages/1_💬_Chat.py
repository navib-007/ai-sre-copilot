"""
1_💬_Chat.py — Multi-Agent SRE Chat Interface
===========================================
CONCEPT: Conversational SRE Copilot
"""

import streamlit as st
import httpx
import uuid
from frontend_utils import inject_custom_css, BACKEND_URL, get_headers, check_login

# 1. Page Config & CSS Ingestion
inject_custom_css()

# 2. Enforce Authentication
if not check_login():
    st.stop()

# 3. Sidebar Profile Info & Session Selector
st.sidebar.markdown("### 👤 Active Session")
st.sidebar.write(f"**User:** {st.session_state['username']}")
st.sidebar.write(f"**Role:** `{st.session_state['role'].upper()}`")

# Fetch sessions list
headers = get_headers()
sessions_list = []
try:
    resp = httpx.get(f"{BACKEND_URL}/api/chat/sessions", headers=headers)
    if resp.status_code == 200:
        sessions_list = resp.json().get("sessions", [])
except Exception as e:
    st.sidebar.error(f"Error loading sessions: {e}")

# Maintain active session ID in state
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# Sidebar Thread Controls
if st.sidebar.button("➕ New Chat Thread", use_container_width=True):
    st.session_state["session_id"] = str(uuid.uuid4())
    st.rerun()

session_ids = [s["session_id"] for s in sessions_list]
if st.session_state["session_id"] not in session_ids:
    dropdown_options = [st.session_state["session_id"]] + session_ids
else:
    dropdown_options = session_ids

if st.session_state["session_id"] in dropdown_options:
    selected_index = dropdown_options.index(st.session_state["session_id"])
else:
    selected_index = 0

selected_session = st.sidebar.selectbox(
    "Select Chat Session",
    dropdown_options,
    index=selected_index,
    format_func=lambda x: f"New Session: {x[:8]}..." if x == st.session_state["session_id"] and x not in session_ids else f"Session: {x[:8]}..."
)

if selected_session != st.session_state["session_id"]:
    st.session_state["session_id"] = selected_session
    st.rerun()

# 4. Main Chat Interface
st.title("💬 Multi-Agent SRE Chat")
st.write("Talk to the SRE Copilot. The supervisor will route your requests to specialized agents for runbooks, tickets, logs, or metrics.")
st.markdown("---")

# Fetch history
chat_history = []
try:
    hist_resp = httpx.get(f"{BACKEND_URL}/api/chat/history/{st.session_state['session_id']}", headers=headers)
    if hist_resp.status_code == 200:
        chat_history = hist_resp.json().get("messages", [])
    elif hist_resp.status_code == 404:
        chat_history = []
except Exception as e:
    st.error(f"Failed to fetch conversation history: {e}")

# Render chat messages
for msg in chat_history:
    role = msg["role"]
    content = msg["content"]
    agent_name = msg.get("agent_name")
    
    with st.chat_message(role):
        st.write(content)
        if role == "assistant" and agent_name:
            st.caption(f"🤖 Specialist: `{agent_name}`")

# Chat input block
if st.session_state["role"] == "viewer":
    st.warning("🔒 Read-Only Mode: Viewers are not authorized to message the AI Agent.")
else:
    user_input = st.chat_input("Ask a question (e.g. 'check frontend service logs', 'escalate a ticket for DB CPU usage', etc.)")
    if user_input:
        # User message
        with st.chat_message("user"):
            st.write(user_input)
            
        # Assistant message with spinner
        with st.chat_message("assistant"):
            with st.spinner("Supervisor Routing & Agent Thinking..."):
                try:
                    payload = {
                        "message": user_input,
                        "session_id": st.session_state["session_id"]
                    }
                    chat_resp = httpx.post(f"{BACKEND_URL}/api/chat/", json=payload, headers=headers, timeout=60.0)
                    
                    if chat_resp.status_code == 200:
                        data = chat_resp.json()
                        response_msg = data.get("message", "")
                        agent_used = data.get("agent_used", "unknown")
                        intent = data.get("intent_detected", "unknown")
                        sources = data.get("sources", [])
                        proc_time = data.get("processing_time_ms", 0.0)
                        
                        st.write(response_msg)
                        st.caption(f"⚡ Intent: `{intent}` ➔ Specialist: `{agent_used}` | Latency: `{proc_time}ms`")
                        
                        if sources:
                            st.markdown("##### 📄 Referenced Runbook Chunks")
                            for src in sources:
                                score_pct = src['score'] * 100
                                st.markdown(
                                    f"""
                                    <div class="citation-box">
                                        <strong>{src['source']}</strong> (Similarity: {score_pct:.1f}%, Chunk: {src['chunk_index']})
                                        <p style="margin: 5px 0 0 0; font-style: italic; color: #a0aabf; font-size: 0.85rem;">
                                            "{src['text_preview']}"
                                        </p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                        
                        # Reload to update sidebar sessions list if new thread
                        if st.session_state["session_id"] not in session_ids:
                            st.rerun()
                    else:
                        st.error(f"Error: {chat_resp.status_code} - {chat_resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")
