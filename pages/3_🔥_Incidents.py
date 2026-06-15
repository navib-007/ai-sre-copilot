"""
3_🔥_Incidents.py — Incident Command Center
==========================================
CONCEPT: Operations Incident Investigation & Management
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

st.title("🔥 Incidents Command Center")
st.write("Log, monitor, and investigate high-priority production outages or system degradations.")
st.markdown("---")

# ─── Incident Reporting Form (Engineers and Admins only) ───────────────────────
if st.session_state["role"] in ["admin", "engineer"]:
    with st.expander("➕ Report New Incident"):
        st.markdown("### Declare a Production Incident")
        i_title = st.text_input("Incident Title", placeholder="e.g. Frontend service throwing 500 errors")
        i_desc = st.text_area("Incident Description / Symptoms", placeholder="e.g. Users report gateway timeout when loading dashboard...")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            i_severity = st.selectbox("Severity Level", ["P1", "P2", "P3", "P4"], index=2)
        with col_c2:
            i_services = st.text_input("Affected Services (comma-separated)", placeholder="e.g. frontend, gateway-api")
            
        if st.button("Declare Incident", use_container_width=True):
            if not i_title or not i_desc:
                st.error("Please fill in both the Incident Title and Description fields.")
            else:
                payload = {
                    "title": i_title,
                    "description": i_desc,
                    "severity": i_severity,
                    "affected_services": i_services if i_services else None
                }
                try:
                    create_resp = httpx.post(f"{BACKEND_URL}/api/incidents/", json=payload, headers=headers)
                    if create_resp.status_code in [200, 201]:
                        st.success(f"Incident reported successfully! Outage tracking activated.")
                        st.rerun()
                    else:
                        st.error(f"Failed to declare incident: {create_resp.status_code} - {create_resp.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")
else:
    st.info("🔒 View-Only Mode: Only engineers and admins can declare incidents.")

# ─── Incident Filtering ───────────────────────────────────────────────────────
st.markdown("### 🔍 Filter Incident Logs")
col_f1, col_f2 = st.columns(2)

with col_f1:
    f_status = st.selectbox("Status", ["All", "detected", "investigating", "mitigated", "resolved"])
with col_f2:
    f_severity = st.selectbox("Severity", ["All", "P1", "P2", "P3", "P4"])

# Build API query parameters
params = {"page": 1, "page_size": 50}
if f_status != "All":
    params["status"] = f_status
if f_severity != "All":
    params["severity"] = f_severity

# Load Incidents
incidents = []
total_incidents = 0
try:
    inc_resp = httpx.get(f"{BACKEND_URL}/api/incidents/", headers=headers, params=params)
    if inc_resp.status_code == 200:
        data = inc_resp.json()
        incidents = data.get("incidents", [])
        total_incidents = data.get("total", 0)
    else:
        st.error(f"Failed to fetch incidents: {inc_resp.status_code} - {inc_resp.text}")
except Exception as e:
    st.error(f"Error communicating with backend: {e}")

st.write(f"Displaying **{len(incidents)}** out of **{total_incidents}** declared incidents.")

# ─── Render Incidents List ────────────────────────────────────────────────────
for i in incidents:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Severity Badge styling
    sev_color = "#b0b0b0"
    if i['severity'] == "P1":
        sev_color = "#ff3333"
    elif i['severity'] == "P2":
        sev_color = "#ff9900"
    elif i['severity'] == "P3":
        sev_color = "#33cc33"
    elif i['severity'] == "P4":
        sev_color = "#3399ff"
        
    status_class = f"badge badge-{i['status'].replace('_', '')}"
    
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 1.15rem; font-weight: 700; color: {sev_color};">🔥 Incident #{i['id']} [{i['severity']}]</span>
            <span class="{status_class}">{i['status']}</span>
        </div>
        <h3 style="margin: 0; padding: 0; font-size: 1.3rem;">{i['title']}</h3>
        <p style="margin: 8px 0; font-size: 0.95rem; color: #d0d7e6;">{i['description']}</p>
        
        <div style="margin: 8px 0; font-size: 0.9rem; padding: 6px; background: rgba(255,255,255,0.02); border-radius: 4px;">
            <strong>Root Cause:</strong> {i['root_cause'] or '<i>Pending root cause analysis...</i>'}
        </div>
        
        <div style="font-size: 0.85rem; color: #a0aabf; margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 8px;">
            <strong>Affected Services:</strong> {i['affected_services'] or 'N/A'} |
            <strong>Reporter ID:</strong> {i['reported_by']} |
            <strong>Detected At:</strong> {i['detected_at']}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Inline update section for engineers and admins
    if st.session_state["role"] in ["admin", "engineer"]:
        with st.expander("🛠️ Update Incident State"):
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                u_status = st.selectbox(
                    "Transition Incident Status",
                    ["detected", "investigating", "mitigated", "resolved"],
                    index=["detected", "investigating", "mitigated", "resolved"].index(i['status']),
                    key=f"status_select_{i['id']}"
                )
            with col_u2:
                u_severity = st.selectbox(
                    "Severity",
                    ["P1", "P2", "P3", "P4"],
                    index=["P1", "P2", "P3", "P4"].index(i['severity']),
                    key=f"severity_select_{i['id']}"
                )
                
            u_services = st.text_input(
                "Affected Services",
                value=i['affected_services'] or "",
                key=f"services_input_{i['id']}"
            )
            u_root_cause = st.text_area(
                "Root Cause Analysis",
                value=i['root_cause'] or "",
                key=f"root_cause_input_{i['id']}",
                placeholder="RCA explanation..."
            )
            
            if st.button("Apply Outage Updates", key=f"apply_{i['id']}", use_container_width=True):
                payload = {}
                if u_status != i['status']:
                    payload["status"] = u_status
                if u_severity != i['severity']:
                    payload["severity"] = u_severity
                if u_services != (i['affected_services'] or ""):
                    payload["affected_services"] = u_services if u_services else None
                if u_root_cause != (i['root_cause'] or ""):
                    payload["root_cause"] = u_root_cause if u_root_cause else None
                    
                if payload:
                    try:
                        patch_resp = httpx.patch(f"{BACKEND_URL}/api/incidents/{i['id']}", json=payload, headers=headers)
                        if patch_resp.status_code == 200:
                            st.success(f"Incident #{i['id']} updated successfully!")
                            st.rerun()
                        else:
                            st.error(f"Failed to update incident: {patch_resp.text}")
                    except Exception as e:
                        st.error(f"Error updating: {e}")
                else:
                    st.info("No modifications specified.")
                    
    st.markdown('</div>', unsafe_allow_html=True)
