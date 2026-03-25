from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="SmartDocs - Multilingual PDF Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.components.upload_panel import render_upload_panel
from ui.components.query_panel import render_query_panel
from ui.components.answer_panel import stream_and_render_answer, render_previous_answer
from ui.components.cost_panel import render_cost_panel

API_BASE = "http://localhost:8000"
DEFAULT_USER_ID = "dev_user_001"


def _init_session_state() -> None:
    defaults = {
        "doc_id": None,
        "doc_title": "your document",
        "doc_language": "en",
        "doc_type": "other",
        "ingestion_status": None,
        "chunks_stored": 0,
        "last_answer": None,
        "last_sources": [],
        "last_metadata": {},
        "last_crag": False,
        "session_cost_inr": 0.0,
        "query_count": 0,
        "prefilled_query": "",
        "user_id": DEFAULT_USER_ID,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()

# ── Reset button ───────────────────────────────────────────────────────────────
if st.sidebar.button("Reset Session", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        color: white;
    ">
        <h1 style="margin:0; font-size:1.8em;">SmartDocs</h1>
        <p style="margin:6px 0 0 0; opacity:0.9; font-size:0.95em;">
            Multilingual PDF Q&A for Indian professionals.
            22 languages. No translation layer.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    with st.expander("User Identity", expanded=False):
        uid_input = st.text_input(
            "User ID",
            value=st.session_state.user_id,
            help="In production this comes from your auth system",
        )
        if uid_input != st.session_state.user_id:
            st.session_state.user_id = uid_input
            st.session_state.doc_id = None
            st.session_state.pop("last_answer", None)
            st.rerun()

user_id = st.session_state.user_id

render_upload_panel(api_base=API_BASE, user_id=user_id)
render_cost_panel()

# ── Main area ──────────────────────────────────────────────────────────────────
if not st.session_state.get("doc_id"):
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "### 1 Upload\n\n"
            "Upload any PDF — GST notices, legal agreements, "
            "insurance policies, government circulars."
        )
    with c2:
        st.markdown(
            "### 2 Detect\n\n"
            "SmartDocs detects the document language automatically. "
            "Hindi, Tamil, Telugu — no setup needed."
        )
    with c3:
        st.markdown(
            "### 3 Ask\n\n"
            "Ask questions in your language. "
            "Get answers in your language. Citations included."
        )
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

# Document is loaded
doc_title = st.session_state.get("doc_title", "your document")
st.markdown(f"**Document:** {doc_title}")
st.markdown("---")

query, language_override, submitted = render_query_panel()
st.markdown("---")

if submitted and query:
    st.session_state.pop("last_answer", None)
    st.session_state.pop("last_sources", None)
    st.session_state.pop("last_metadata", None)
    stream_and_render_answer(
        query=query,
        doc_id=st.session_state.doc_id,
        doc_title=doc_title,
        api_base=API_BASE,
        user_id=user_id,
    )
elif st.session_state.get("last_answer"):
    render_previous_answer()
else:
    st.info("Ask a question above to get started.")