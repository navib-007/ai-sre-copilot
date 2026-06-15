"""
5_⚙️_Admin.py — Remediation Approvals Dashboard
================================================
CONCEPT: Human-in-the-Loop Remediation Gates
"""

import streamlit as st
import httpx
from frontend_utils import inject_custom_css, BACKEND_URL, get_headers, check_login

# 1. Page Config & CSS Ingestion
inject_custom_css()

# 2. Enforce Authentication
if not check_login():
    st.stop()

# 3. Sidebar Profile Info
st.sidebar.markdown("### 👤 Active Session")
st.sidebar.write(f"**User:** {st.session_state['username']}")
st.sidebar.write(f"**Role:** `{st.session_state['role'].upper()}`")

headers = get_headers()

# 4. Enforce Page Authorization (Engineers & Admins only)
if st.session_state["role"] not in ["admin", "engineer"]:
    st.warning("🔒 Access Denied. The Remediation Approvals page is restricted to engineers and administrators.")
    st.info("Your current role is `viewer`. Viewers do not have access to approvals control.")
    st.stop()

st.title("⚙️ Remediation Approvals Dashboard")
st.write("Review, audit, approve, or reject high-risk automated remediation tasks proposed by SRE specialist agents.")
st.markdown("---")

# Fetch pending approvals list
pending_requests = []
try:
    pending_resp = httpx.get(f"{BACKEND_URL}/api/approvals/pending", headers=headers)
    if pending_resp.status_code == 200:
        pending_requests = pending_resp.json()
    else:
        st.error(f"Failed to fetch pending approvals: {pending_resp.status_code} - {pending_resp.text}")
except Exception as e:
    st.error(f"Error communicating with backend: {e}")

st.write(f"There are currently **{len(pending_requests)}** pending remediation approvals in the queue.")

# Render pending approvals
for req in pending_requests:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 1.15rem; font-weight: 700; color: #00d2ff;">⚙️ Request #{req['id']}</span>
            <span class="badge badge-progress">{req['status'].upper()}</span>
        </div>
        <div>
            <strong>Action Type:</strong> <code>{req['action_type']}</code> <br/>
            <strong>Incident ID Reference:</strong> {req['incident_id'] or 'N/A'} <br/>
            <strong>LangGraph Session ID (Thread):</strong> <code>{req['langgraph_thread_id'] or 'N/A'}</code> <br/>
            <strong>Requested By User ID:</strong> {req['requested_by']} | <strong>Created At:</strong> {req['created_at']}
        </div>
        <div style="margin-top: 10px; margin-bottom: 10px;">
            <strong>Action Details:</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.json(req.get("action_details", {}))
    
    # Render interactive controls based on user role
    if st.session_state["role"] == "admin":
        st.markdown("<hr style='margin: 10px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
        comment = st.text_input(
            "Justification / Comment",
            placeholder="e.g. Approved. Confirmed CPU metrics spike.",
            key=f"comment_{req['id']}"
        )
        
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("✅ Approve Action", key=f"approve_{req['id']}", use_container_width=True):
                try:
                    payload = {"comment": comment if comment else None}
                    action_resp = httpx.post(f"{BACKEND_URL}/api/approvals/{req['id']}/approve", json=payload, headers=headers)
                    if action_resp.status_code == 200:
                        st.success(f"Action #{req['id']} successfully APPROVED!")
                        st.rerun()
                    else:
                        st.error(f"Failed to approve request: {action_resp.status_code} - {action_resp.text}")
                except Exception as e:
                    st.error(f"Error processing approval: {e}")
        with col_act2:
            if st.button("❌ Reject Action", key=f"reject_{req['id']}", use_container_width=True):
                try:
                    payload = {"comment": comment if comment else None}
                    action_resp = httpx.post(f"{BACKEND_URL}/api/approvals/{req['id']}/reject", json=payload, headers=headers)
                    if action_resp.status_code == 200:
                        st.warning(f"Action #{req['id']} successfully REJECTED.")
                        st.rerun()
                    else:
                        st.error(f"Failed to reject request: {action_resp.status_code} - {action_resp.text}")
                except Exception as e:
                    st.error(f"Error processing rejection: {e}")
    else:
        st.info("🔒 Review-Only: Only administrators can approve/reject pending remediation actions.")
        
    st.markdown('</div>', unsafe_allow_html=True)
