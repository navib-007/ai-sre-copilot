"""
4_📄_Documents.py — Knowledge Base runbooks
===========================================
CONCEPT: Semantic RAG Search and Ingestion pipeline
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

st.title("📄 Knowledge Base Documents")
st.write("Perform semantic search across incident runbooks, or manage files in the RAG pipeline vector database.")
st.markdown("---")

# ─── SEMANTIC SEARCH (ALL ROLES) ──────────────────────────────────────────────
st.markdown("## 🔍 Semantic Similarity Search")
search_query = st.text_input("Enter Search Term / Query", placeholder="e.g. How do I fix a database crash?")

col_s1, col_s2 = st.columns(2)
with col_s1:
    top_k = st.slider("Top K (Number of Results)", min_value=1, max_value=15, value=5)
with col_s2:
    threshold = st.slider("Relevance Score Threshold", min_value=0.0, max_value=1.0, value=0.55, step=0.05)

if search_query:
    with st.spinner("Embedding query & searching vector database..."):
        try:
            payload = {
                "query": search_query,
                "top_k": top_k,
                "score_threshold": threshold
            }
            search_resp = httpx.post(f"{BACKEND_URL}/api/documents/search", json=payload, headers=headers)
            
            if search_resp.status_code == 200:
                results_data = search_resp.json()
                results = results_data.get("results", [])
                st.write(f"Found **{len(results)}** matching runbook chunks (Search Latency: `{results_data.get('search_time_ms', 0.0)}ms`):")
                
                for r in results:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    score_pct = r['score'] * 100
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 700; color: #00d2ff;">📄 {r['source']}</span>
                            <span class="badge badge-resolved">Score: {score_pct:.1f}%</span>
                        </div>
                        <p style="margin: 8px 0; font-size: 0.95rem; color: #ecf0f1; font-family: monospace; white-space: pre-wrap;">{r['text']}</p>
                        <div style="font-size: 0.8rem; color: #a0aabf; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 6px;">
                            <strong>Document ID:</strong> {r['document_id']} | <strong>Chunk Index:</strong> {r['chunk_index']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error(f"Search request failed: {search_resp.status_code} - {search_resp.text}")
        except Exception as e:
            st.error(f"Error executing semantic search: {e}")

st.markdown("---")

# ─── ADMIN ONLY CONTROLS (UPLOAD, LIST, CACHE STATS) ─────────────────────────
if st.session_state["role"] == "admin":
    st.markdown("## ⚙️ Vector Database Management (Admin Only)")
    
    tab_upload, tab_cache, tab_docs = st.tabs(["📤 Upload & Ingest", "💾 Embedding Cache Stats", "🗂️ Ingested Documents List"])
    
    # ── TAB 1: Document Upload Ingestion ──
    with tab_upload:
        st.subheader("Upload Document to RAG Ingestion Pipeline")
        uploaded_file = st.file_uploader(
            "Select text, markdown, pdf, or docx file",
            type=["txt", "md", "pdf", "docx"]
        )
        
        if uploaded_file is not None:
            if st.button("Submit to Ingestion Pipeline", use_container_width=True):
                with st.spinner("Parsing, chunking, embedding, and indexing document..."):
                    try:
                        # Construct multipart/form-data payload
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        upload_resp = httpx.post(f"{BACKEND_URL}/api/documents/upload", files=files, headers=headers)
                        
                        if upload_resp.status_code in [200, 201]:
                            result = upload_resp.json()
                            st.success(f"Successfully processed document: `{uploaded_file.name}`")
                            
                            # Display metrics cards
                            st.markdown("### Ingestion Summary")
                            mc1, mc2, mc3, mc4 = st.columns(4)
                            with mc1:
                                st.metric("Chunks Created", result.get("chunks_processed", 0))
                            with mc2:
                                st.metric("Total Tokens", result.get("total_tokens", 0))
                            with mc3:
                                st.metric("Embedding Cache Hits", result.get("cache_hits", 0))
                            with mc4:
                                st.metric("OpenAI API Calls", result.get("api_calls_made", 0))
                        else:
                            st.error(f"Ingestion failed: {upload_resp.status_code} - {upload_resp.text}")
                    except Exception as e:
                        st.error(f"Failed to communicate with ingestion engine: {e}")
                        
    # ── TAB 2: Embedding Cache Statistics ──
    with tab_cache:
        st.subheader("Embedding Cache Statistics Dashboard")
        
        try:
            stats_resp = httpx.get(f"{BACKEND_URL}/api/documents/cache/stats", headers=headers)
            if stats_resp.status_code == 200:
                stats = stats_resp.json()
                
                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1:
                    st.metric("L1 (In-Memory) Size", stats.get("memory_cache_size", 0))
                with sc2:
                    st.metric("L1 Cache Hits", stats.get("memory_hits", 0))
                with sc3:
                    st.metric("L2 (Database) Hits", stats.get("db_hits", 0))
                with sc4:
                    st.metric("Total Hit Rate", f"{stats.get('hit_rate_percent', 0.0):.1f}%")
                
                # Show absolute values
                st.write(f"💾 **Total Requests Served:** {stats.get('total_requests', 0)} | 🤖 **OpenAI API Calls Made:** {stats.get('api_calls', 0)}")
                
                if st.button("🧹 Clear In-Memory L1 Cache", use_container_width=True):
                    clear_resp = httpx.delete(f"{BACKEND_URL}/api/documents/cache/clear", headers=headers)
                    if clear_resp.status_code == 200:
                        st.success("L1 Cache flushed successfully!")
                        st.rerun()
                    else:
                        st.error(f"Failed to clear cache: {clear_resp.text}")
            else:
                st.error(f"Failed to fetch cache stats: {stats_resp.status_code}")
        except Exception as e:
            st.error(f"Error fetching cache stats: {e}")
            
    # ── TAB 3: Ingested Documents List and Deletion ──
    with tab_docs:
        st.subheader("Ingested Documents in SQLite/Qdrant")
        
        try:
            list_resp = httpx.get(f"{BACKEND_URL}/api/documents/", headers=headers)
            if list_resp.status_code == 200:
                docs_data = list_resp.json().get("documents", [])
                
                if not docs_data:
                    st.info("No documents have been ingested yet.")
                else:
                    for d in docs_data:
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.markdown(
                            f"""
                            <strong>ID:</strong> {d['id']} | <strong>Filename:</strong> `{d['filename']}` <br/>
                            <strong>Type:</strong> {d['file_type'].upper()} | <strong>Chunks:</strong> {d['chunk_count']} | <strong>Uploaded:</strong> {d['uploaded_at']}
                            """,
                            unsafe_allow_html=True
                        )
                        # Deletion Button
                        if st.button(f"🗑️ Delete Document #{d['id']}", key=f"delete_doc_{d['id']}", use_container_width=True):
                            with st.spinner("Deleting document from SQLite and removing vectors from Qdrant..."):
                                del_resp = httpx.delete(f"{BACKEND_URL}/api/documents/{d['id']}", headers=headers)
                                if del_resp.status_code in [200, 204]:
                                    st.success(f"Document #{d['id']} deleted successfully!")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to delete document: {del_resp.text}")
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error(f"Failed to fetch document list: {list_resp.status_code}")
        except Exception as e:
            st.error(f"Error fetching documents: {e}")
else:
    st.info("🔒 Management Panel: Only users with the `admin` role can upload documents, clear embedding cache, or delete vector collections.")
