"""
2_🎫_Tickets.py — Support Ticket Board
======================================
CONCEPT: Ticket CRUD and Lifecycle Management
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

st.title("🎫 Helpdesk Ticket Board")
st.write("View, filter, create, and transition support tickets in the SRE environment.")
st.markdown("---")

# ─── Ticket Creation Form (Engineers and Admins only) ─────────────────────────
if st.session_state["role"] in ["admin", "engineer"]:
    with st.expander("➕ Create New Ticket"):
        st.markdown("### Report a New System Issue")
        t_title = st.text_input("Ticket Title", placeholder="e.g. Memory leak in order service")
        t_desc = st.text_area("Detailed Description", placeholder="e.g. Logs show heap consumption growing monotonically...")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            t_priority = st.selectbox("Priority Level", ["low", "medium", "high", "critical"], index=1)
        with col_c2:
            t_category = st.selectbox("Category", ["database", "kubernetes", "network", "application", "hardware", "other"], index=3)
            
        t_tags_raw = st.text_input("Tags (comma-separated)", placeholder="e.g. prod, backend, memory")
        
        if st.button("Submit Ticket", use_container_width=True):
            if not t_title or not t_desc:
                st.error("Please fill in both the Title and Description fields.")
            else:
                tags = [t.strip() for t in t_tags_raw.split(",") if t.strip()] if t_tags_raw else []
                payload = {
                    "title": t_title,
                    "description": t_desc,
                    "priority": t_priority,
                    "category": t_category,
                    "tags": tags
                }
                try:
                    create_resp = httpx.post(f"{BACKEND_URL}/api/tickets/", json=payload, headers=headers)
                    if create_resp.status_code in [200, 201]:
                        st.success("Ticket successfully submitted!")
                        st.rerun()
                    else:
                        st.error(f"Failed to submit ticket: {create_resp.status_code} - {create_resp.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")
else:
    st.info("🔒 View-Only Mode: Only engineers and admins can create tickets.")

# ─── Ticket Filtering ─────────────────────────────────────────────────────────
st.markdown("### 🔍 Filter and Search Tickets")
col_f1, col_f2, col_f3, col_f4 = st.columns(4)

with col_f1:
    f_status = st.selectbox("Status", ["All", "open", "in_progress", "resolved", "closed"])
with col_f2:
    f_priority = st.selectbox("Priority", ["All", "low", "medium", "high", "critical"])
with col_f3:
    f_category = st.selectbox("Category", ["All", "database", "kubernetes", "network", "application", "hardware", "other"])
with col_f4:
    f_search = st.text_input("Search Text", placeholder="Search in tickets...")

# Build API request parameters
params = {"page": 1, "page_size": 50}
if f_status != "All":
    params["status"] = f_status
if f_priority != "All":
    params["priority"] = f_priority
if f_category != "All":
    params["category"] = f_category
if f_search:
    params["search"] = f_search

# Load Tickets
tickets = []
total_tickets = 0
try:
    tickets_resp = httpx.get(f"{BACKEND_URL}/api/tickets/", headers=headers, params=params)
    if tickets_resp.status_code == 200:
        data = tickets_resp.json()
        tickets = data.get("tickets", [])
        total_tickets = data.get("total", 0)
    else:
        st.error(f"Failed to fetch tickets: {tickets_resp.status_code} - {tickets_resp.text}")
except Exception as e:
    st.error(f"Error communicating with backend: {e}")

st.write(f"Showing **{len(tickets)}** out of **{total_tickets}** tickets matching current filters.")

# ─── Render Tickets List ──────────────────────────────────────────────────────
for t in tickets:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Status styling mapping
    status_class = f"badge badge-{t['status'].replace('_', '')}"
    
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 1.15rem; font-weight: 700; color: #00d2ff;">🎫 Ticket #{t['id']}</span>
            <span class="{status_class}">{t['status']}</span>
        </div>
        <h3 style="margin: 0; padding: 0; font-size: 1.3rem;">{t['title']}</h3>
        <p style="margin: 8px 0; font-size: 0.95rem; color: #d0d7e6;">{t['description']}</p>
        <div style="font-size: 0.85rem; color: #a0aabf; margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 8px;">
            <strong>Priority:</strong> <span style="text-transform: capitalize;">{t['priority']}</span> |
            <strong>Category:</strong> <span style="text-transform: capitalize;">{t['category'] or 'N/A'}</span> |
            <strong>Creator ID:</strong> {t['created_by']} |
            <strong>Assignee ID:</strong> {t['assigned_to'] or 'Unassigned'}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Inline update section for authorized roles
    if st.session_state["role"] in ["admin", "engineer"]:
        with st.expander("🛠️ Update Ticket State"):
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                u_status = st.selectbox(
                    "Transition Status",
                    ["open", "in_progress", "resolved", "closed"],
                    index=["open", "in_progress", "resolved", "closed"].index(t['status']),
                    key=f"status_select_{t['id']}"
                )
            with col_u2:
                u_assignee = st.number_input(
                    "Assignee User ID",
                    value=t['assigned_to'] or 0,
                    step=1,
                    key=f"assignee_select_{t['id']}"
                )
                
            if st.button("Apply Changes", key=f"apply_{t['id']}", use_container_width=True):
                payload = {}
                if u_status != t['status']:
                    payload["status"] = u_status
                if (u_assignee > 0 and u_assignee != t['assigned_to']) or (u_assignee == 0 and t['assigned_to'] is not None):
                    payload["assigned_to"] = u_assignee if u_assignee > 0 else None
                
                if payload:
                    try:
                        patch_resp = httpx.patch(f"{BACKEND_URL}/api/tickets/{t['id']}", json=payload, headers=headers)
                        if patch_resp.status_code == 200:
                            st.success(f"Ticket #{t['id']} updated successfully!")
                            st.rerun()
                        else:
                            st.error(f"Failed to update ticket: {patch_resp.text}")
                    except Exception as e:
                        st.error(f"Error updating: {e}")
                else:
                    st.info("No modifications specified.")
                    
    st.markdown('</div>', unsafe_allow_html=True)
