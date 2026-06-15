"""
streamlit_app.py — Landing Page, Login, and Welcome Dashboard
=============================================================
CONCEPT: Frontend Application Entry Point
"""

import streamlit as st
import httpx
from frontend_utils import inject_custom_css, BACKEND_URL, get_headers

# Inject styling config
inject_custom_css()

# ─── Sidebar User Profile ─────────────────────────────────────────────────────
if "token" in st.session_state:
    st.sidebar.markdown(f"### 👤 Active Session")
    st.sidebar.write(f"**User:** {st.session_state['username']}")
    st.sidebar.write(f"**Role:** `{st.session_state['role'].upper()}`")
    
    if st.sidebar.button("Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ─── Login Screen ─────────────────────────────────────────────────────────────
if "token" not in st.session_state:
    st.markdown(
        """
        <div style="text-align: center; margin-top: 50px; margin-bottom: 30px;">
            <h1 style="font-size: 3rem;">🤖 Agentic AI Ops Platform</h1>
            <p style="font-size: 1.2rem; color: #a0aabf;">Your Pair-Programming SRE Copilot and Incident Remediation Assistant</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🔑 Authenticate to Continue")
        
        username = st.text_input("Username", placeholder="e.g. admin or alice_engineer")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        
        if st.button("Log In", use_container_width=True):
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                try:
                    # POST urlencoded form data to backend login endpoint
                    login_data = {"username": username, "password": password}
                    login_resp = httpx.post(f"{BACKEND_URL}/api/auth/login", data=login_data)
                    
                    if login_resp.status_code == 200:
                        token_info = login_resp.json()
                        st.session_state["token"] = token_info["access_token"]
                        st.session_state["username"] = username
                        
                        # Load user profile profile details to fetch role info
                        headers = {"Authorization": f"Bearer {token_info['access_token']}"}
                        me_resp = httpx.get(f"{BACKEND_URL}/api/auth/me", headers=headers)
                        
                        if me_resp.status_code == 200:
                            profile = me_resp.json()
                            st.session_state["role"] = profile["role"]
                            st.session_state["user_id"] = profile["id"]
                            st.success(f"Welcome back, {username}!")
                            st.rerun()
                        else:
                            st.error("Authentication completed but profile lookup failed.")
                    else:
                        st.error("Invalid username or password.")
                except Exception as e:
                    st.error(f"Failed to communicate with backend server: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Admin instructions helper box
        st.info(
            "💡 **Default seeded credentials**:\n\n"
            "- **Admin User**: `admin` / `secret` (Full privileges)\n"
            "- **Engineer User**: `alice_engineer` / `secret` (Read/Write, Chat & Tickets)\n"
            "- **Viewer User**: `bob_viewer` / `secret` (Read-only, metrics & ticket list)"
        )

# ─── Logged In Dashboard ──────────────────────────────────────────────────────
else:
    st.markdown(f"# Welcome back, {st.session_state['username']}! 👋")
    st.write("Here is the live status dashboard of the SRE Copilot Platform environment.")
    st.markdown("---")
    
    # 1. Load data from backend
    health_status = "Unknown"
    db_status = "Disconnected"
    uptime_sec = 0.0
    open_tickets_count = 0
    pending_approvals_count = 0
    
    try:
        # Load Health
        health_resp = httpx.get(f"{BACKEND_URL}/health")
        if health_resp.status_code == 200:
            h_data = health_resp.json()
            health_status = h_data.get("status", "healthy").upper()
            db_status = h_data.get("database", "disconnected").upper()
            uptime_sec = h_data.get("uptime_seconds", 0.0)
            
        # Load Tickets Count (secured - requires token)
        headers = get_headers()
        tickets_resp = httpx.get(f"{BACKEND_URL}/api/tickets/", headers=headers)
        if tickets_resp.status_code == 200:
            t_data = tickets_resp.json()
            # Count only 'open' or 'in_progress' tickets
            open_tickets_count = sum(
                1 for t in t_data.get("tickets", []) if t.get("status") in ["open", "in_progress"]
            )
            
        # Load Approvals Count
        if st.session_state["role"] in ["admin", "engineer"]:
            approvals_resp = httpx.get(f"{BACKEND_URL}/api/approvals/pending", headers=headers)
            if approvals_resp.status_code == 200:
                pending_approvals_count = len(approvals_resp.json())
    except Exception as e:
        st.warning(f"Error fetching live metrics from backend: {str(e)}")
        
    # 2. Main Metrics Grid
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.metric("System Health", health_status, delta="Operational" if health_status == "HEALTHY" else "Degraded")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with m_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.metric("Database Status", db_status, delta="Active" if db_status == "CONNECTED" else "Inactive")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with m_col3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.metric("Active Tickets", open_tickets_count, delta_color="inverse")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with m_col4:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.metric("Pending Approvals", pending_approvals_count, delta_color="inverse")
        st.markdown("</div>", unsafe_allow_html=True)

    # 3. Main Dashboard Layout
    layout_col1, layout_col2 = st.columns([2, 1.2])
    
    with layout_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("💡 SRE Multi-Agent Copilot Capabilities")
        st.write(
            "The platform leverages a specialized LangGraph multi-agent network orchestrated "
            "by a Supervisor Agent to assist with site reliability engineering operations. Use "
            "the sidebar navigation features to interact with:"
        )
        
        st.markdown(
            """
            - **💬 AI Agent Chat**: Talk to the SRE Copilot. It retrieves runbooks from the knowledge base,
              checks simulated log systems and metrics (Four Golden Signals), opens or escalates tickets, and
              suggests remediation scripts.
            - **🎫 Helpdesk Tickets**: View the live support ticket board, report new system bugs, and transition resolutions.
            - **🔥 Incidents Command Center**: Log and track P1/P2 operational incident histories.
            - **📄 Knowledge Base Documents**: Upload runbooks, search technical procedures, and view vector indexes.
            - **⚙️ Remediation Approvals**: Grant or deny approvals for high-risk actions (e.g. restarting a pod) proposed by the agent.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
    with layout_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 Uptime Statistics")
        days = int(uptime_sec // 86400)
        hours = int((uptime_sec % 86400) // 3600)
        minutes = int((uptime_sec % 3600) // 60)
        seconds = int(uptime_sec % 60)
        
        st.write(f"⏱️ **Process Uptime:** {days}d {hours}h {minutes}m {seconds}s")
        st.progress(min(1.0, uptime_sec / 3600.0), text="System Health Score: 100%")
        st.markdown("</div>", unsafe_allow_html=True)
