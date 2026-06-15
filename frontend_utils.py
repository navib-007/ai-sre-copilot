"""
frontend_utils.py — Shared Styling and Authentication Helpers
=============================================================
CONCEPT: Clean modular helper functions for the Streamlit frontend.
"""

import streamlit as st

BACKEND_URL = "http://localhost:8000"


def inject_custom_css():
    """Sets page config and injects premium CSS styling (glassmorphism & dark theme)."""
    # 1. Page Config
    st.set_page_config(
        page_title="AI SRE Copilot Platform",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 2. Curated Premium Dark Mode CSS
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;800&display=swap');
        
        /* Apply fonts */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #0b0f19;
            color: #ecf0f1;
        }

        /* Title styling with smooth modern gradients */
        h1, h2, h3 {
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, #00d2ff 0%, #0072ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            margin-bottom: 0.8rem;
        }

        /* Glassmorphic card custom element */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(12px);
            margin-bottom: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .glass-card:hover {
            border-color: rgba(0, 210, 255, 0.3);
            transform: translateY(-2px);
        }

        /* Metric values formatting */
        [data-testid="stMetricValue"] {
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            background: linear-gradient(135deg, #00ffcc 0%, #0099ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Premium custom gradient buttons */
        .stButton>button {
            background: linear-gradient(135deg, #00d2ff 0%, #0072ff 100%) !important;
            color: #ffffff !important;
            border: none !important;
            padding: 10px 28px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            border-radius: 8px !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            box-shadow: 0 4px 14px 0 rgba(0, 114, 255, 0.4) !important;
        }
        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px 0 rgba(0, 114, 255, 0.6) !important;
            background: linear-gradient(135deg, #0072ff 0%, #00d2ff 100%) !important;
        }

        /* Badge badges */
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .badge-open {
            background-color: rgba(255, 75, 75, 0.12);
            color: #ff4b4b;
            border: 1px solid rgba(255, 75, 75, 0.25);
        }
        .badge-progress {
            background-color: rgba(255, 165, 0, 0.12);
            color: #ffa500;
            border: 1px solid rgba(255, 165, 0, 0.25);
        }
        .badge-resolved {
            background-color: rgba(9, 171, 59, 0.12);
            color: #09ab3b;
            border: 1px solid rgba(9, 171, 59, 0.25);
        }
        .badge-closed {
            background-color: rgba(140, 140, 140, 0.12);
            color: #b0b0b0;
            border: 1px solid rgba(140, 140, 140, 0.25);
        }

        /* Customize sidebar style */
        [data-testid="stSidebar"] {
            background-color: #080b11;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Chat source citations container */
        .citation-box {
            background-color: rgba(255, 255, 255, 0.02);
            border-left: 3px solid #00d2ff;
            padding: 10px 15px;
            border-radius: 4px;
            margin-top: 10px;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def check_login() -> bool:
    """
    Validates if the user is authenticated. 
    If not, blocks the page execution and renders instructions.
    """
    if "token" not in st.session_state:
        st.warning("🔒 Access Denied. Please log in on the Main Page.")
        st.info("👈 Use the Sidebar to navigate back to the main landing page to authenticate.")
        return False
    return True


def get_headers() -> dict:
    """Generate headers containing the logged-in user's Bearer JWT token."""
    if "token" in st.session_state:
        return {"Authorization": f"Bearer {st.session_state['token']}"}
    return {}
