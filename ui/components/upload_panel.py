# smartdocs/ui/components/upload_panel.py
"""
WHY THIS EXISTS:
Handles PDF upload and displays the document language badge (LAW 17).
Language badge is the product's first impression — shown before the user
types a single query. Hindi PDF → "🇮🇳 Hindi". Bilingual → "🇮🇳 Hindi + 🇬🇧 English".

DESIGN:
  - Upload triggers POST /ingest immediately
  - Language badge rendered from primary_language in ingest response
  - doc_id stored in session_state for all downstream query calls
  - Previously uploaded documents listed from GET /ingest/documents
  - Idempotent: same PDF uploaded twice → returns existing doc_id silently
"""

from __future__ import annotations

import streamlit as st
import httpx
from pathlib import Path

# Language badge map — matches language_detector.py LANGUAGE_BADGE
LANGUAGE_BADGES = {
    "hi": "🇮🇳 Hindi",
    "en": "🇬🇧 English",
    "ta": "🇮🇳 Tamil",
    "te": "🇮🇳 Telugu",
    "kn": "🇮🇳 Kannada",
    "ml": "🇮🇳 Malayalam",
    "mr": "🇮🇳 Marathi",
    "gu": "🇮🇳 Gujarati",
    "pa": "🇮🇳 Punjabi",
    "bn": "🇮🇳 Bengali",
    "ur": "🇮🇳 Urdu",
}

DOC_TYPE_LABELS = {
    "gst_notice": "📋 GST Notice",
    "legal_agreement": "⚖️ Legal Agreement",
    "insurance_policy": "🛡️ Insurance Policy",
    "government_circular": "🏛️ Government Circular",
    "tax_notice": "📑 Tax Notice",
    "other": "📄 Document",
}


def _get_language_badge(lang_code: str) -> str:
    return LANGUAGE_BADGES.get(lang_code, f"🌐 {lang_code.upper()}")


def _get_doc_type_label(doc_type: str) -> str:
    return DOC_TYPE_LABELS.get(doc_type, "📄 Document")


def _ingest_file(
    file_bytes: bytes,
    file_name: str,
    doc_title: str,
    api_base: str,
    user_id: str,
) -> dict:
    """
    Calls POST /ingest with the uploaded file bytes.
    Returns the API response dict.
    Raises on network error — caller handles display.
    """
    response = httpx.post(
        f"{api_base}/ingest",
        headers={"X-User-ID": user_id},
        files={"file": (file_name, file_bytes, "application/pdf")},
        data={"doc_title": doc_title} if doc_title else {},
        timeout=180.0,  # Large PDFs can take up to 3 minutes on CPU
    )
    response.raise_for_status()
    return response.json()


def _load_user_documents(api_base: str, user_id: str) -> list[dict]:
    """Fetches all documents for this user from GET /ingest/documents."""
    try:
        r = httpx.get(
            f"{api_base}/ingest/documents",
            headers={"X-User-ID": user_id},
            timeout=15.0,
        )
        r.raise_for_status()
        return r.json().get("documents", [])
    except Exception:
        return []


def render_upload_panel(api_base: str, user_id: str) -> None:
    """
    Renders the complete upload panel in the Streamlit sidebar.

    State written to st.session_state:
      - doc_id: str — active document UUID for queries
      - doc_title: str — display title
      - doc_language: str — ISO language code
      - doc_type: str — gst_notice | legal | etc.
      - ingestion_status: str — "ready" | "processing" | "failed"
    """
    st.sidebar.markdown("## 📁 Document")

    # ── Previously uploaded documents ─────────────────────────────────────────
    existing_docs = _load_user_documents(api_base, user_id)

    if existing_docs:
        st.sidebar.markdown("**Select existing document:**")
        doc_options = {
            f"{_get_doc_type_label(d['doc_type'])} — {d['title']} ({d['file_name']})": d
            for d in existing_docs
            if d["ingestion_status"] == "completed"
        }

        if doc_options:
            # Pre-select the currently active document if set
            current_label = None
            if st.session_state.get("doc_id"):
                for label, d in doc_options.items():
                    if d["doc_id"] == st.session_state.get("doc_id"):
                        current_label = label
                        break

            selected_label = st.sidebar.selectbox(
                "Choose document",
                options=list(doc_options.keys()),
                index=list(doc_options.keys()).index(current_label)
                if current_label
                else 0,
                label_visibility="collapsed",
            )

            selected_doc = doc_options[selected_label]

            # Update session state when selection changes
            if st.session_state.get("doc_id") != selected_doc["doc_id"]:
                st.session_state.doc_id = selected_doc["doc_id"]
                st.session_state.doc_title = selected_doc["title"]
                st.session_state.doc_language = selected_doc["primary_language"]
                st.session_state.doc_type = selected_doc["doc_type"]
                st.session_state.ingestion_status = "ready"
                # Clear previous query results when switching documents
                st.session_state.pop("last_answer", None)
                st.session_state.pop("last_sources", None)
                st.session_state.pop("last_metadata", None)

        st.sidebar.markdown("---")

    # ── Upload new document ────────────────────────────────────────────────────
    st.sidebar.markdown("**Upload new PDF:**")

    uploaded_file = st.sidebar.file_uploader(
        "Choose PDF",
        type=["pdf"],
        help="GST notices, legal agreements, insurance policies, government circulars",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        doc_title_input = st.sidebar.text_input(
            "Document title (optional)",
            placeholder=Path(uploaded_file.name).stem,
            help="Leave blank to use filename as title",
        )

        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            ingest_clicked = st.button(
                "📤 Upload & Ingest",
                use_container_width=True,
                type="primary",
            )

        if ingest_clicked:
            file_bytes = uploaded_file.read()
            title = doc_title_input.strip() or Path(uploaded_file.name).stem

            with st.sidebar.status("Ingesting document...", expanded=True) as status:
                st.write("📖 Extracting text...")
                try:
                    result = _ingest_file(
                        file_bytes=file_bytes,
                        file_name=uploaded_file.name,
                        doc_title=title,
                        api_base=api_base,
                        user_id=user_id,
                    )

                    if result.get("success") or result.get("ingestion_skipped"):
                        st.write("✂️ Chunking & embedding...")
                        st.write("💾 Stored in vector database")

                        st.session_state.doc_id = result["doc_id"]
                        st.session_state.doc_title = result.get("title", title)
                        st.session_state.doc_language = result.get("primary_language", "en")
                        st.session_state.doc_type = result.get("doc_type", "other")
                        st.session_state.ingestion_status = "ready"
                        st.session_state.chunks_stored = result.get("chunks_stored", 0)
                        st.session_state.pop("last_answer", None)
                        st.session_state.pop("last_sources", None)
                        st.session_state.pop("last_metadata", None)

                        status.update(label="✅ Document ready!", state="complete")
                    else:
                        status.update(label="❌ Ingestion failed", state="error")
                        st.error(result.get("error", "Unknown error"))

                except httpx.TimeoutException:
                    status.update(label="⏱️ Timeout", state="error")
                    st.error(
                        "Upload timed out. Large PDFs can take up to 3 minutes on CPU. "
                        "Try again."
                    )
                except Exception as e:
                    status.update(label="❌ Error", state="error")
                    st.error(f"Upload failed: {str(e)[:200]}")

            st.rerun()

    # ── Active document info + language badge ──────────────────────────────────
    # LAW 17: Language badge shown before user types a single query
    if st.session_state.get("doc_id"):
        lang_code = st.session_state.get("doc_language", "en")
        doc_type = st.session_state.get("doc_type", "other")
        badge = _get_language_badge(lang_code)
        type_label = _get_doc_type_label(doc_type)

        st.sidebar.markdown("---")
        st.sidebar.markdown("**Active document:**")

        # Language badge — the product's first impression
        st.sidebar.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 8px;
                padding: 12px 16px;
                margin: 8px 0;
                color: white;
            ">
                <div style="font-size: 1.1em; font-weight: 600;">{badge}</div>
                <div style="font-size: 0.85em; opacity: 0.9; margin-top: 4px;">{type_label}</div>
                <div style="font-size: 0.75em; opacity: 0.7; margin-top: 2px; font-family: monospace;">
                    {st.session_state.get('doc_id', '')[:8]}...
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        chunks = st.session_state.get("chunks_stored", 0)
        if chunks:
            st.sidebar.caption(f"📦 {chunks} chunks indexed")
