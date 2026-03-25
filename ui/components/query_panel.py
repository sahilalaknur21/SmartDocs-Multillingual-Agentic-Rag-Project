# smartdocs/ui/components/query_panel.py
"""
WHY THIS EXISTS:
Renders the query input box and language selector dropdown.
Language selector lets the user override auto-detection — useful when
asking in a different language than the document (e.g. Tamil doc, English query).

Per spec: "query input + language selector dropdown"
"""

from __future__ import annotations

import streamlit as st

# Languages available in the dropdown
# ISO code → display name
SUPPORTED_LANGUAGES = {
    "auto": "🔍 Auto-detect",
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

# Example queries shown as quick-select chips
EXAMPLE_QUERIES = {
    "en": [
        "What is the total demand amount?",
        "What section of the law applies here?",
        "Summarize this document",
        "What are the key dates mentioned?",
    ],
    "hi": [
        "इस दस्तावेज़ में कुल मांग राशि क्या है?",
        "यहाँ कौन सी धारा लागू होती है?",
        "इस नोटिस का सारांश बताइए",
        "मुख्य तिथियाँ क्या हैं?",
    ],
}


def render_query_panel() -> tuple[str, str, bool]:
    """
    Renders the query input section.

    Returns:
        tuple of:
          - query: str — the user's query text (empty string if not submitted)
          - language_override: str — "auto" or ISO code
          - submitted: bool — True if user clicked Ask or pressed Enter
    """
    # ── Language selector ──────────────────────────────────────────────────────
    lang_col, _ = st.columns([1, 3])
    with lang_col:
        language_override = st.selectbox(
            "Response language",
            options=list(SUPPORTED_LANGUAGES.keys()),
            format_func=lambda k: SUPPORTED_LANGUAGES[k],
            index=0,  # Default: auto-detect
            help=(
                "Auto-detect reads your query's language automatically. "
                "Override if you want answers in a specific language."
            ),
        )

    # ── Example queries (shown before first query) ─────────────────────────────
    if not st.session_state.get("last_answer"):
        doc_lang = st.session_state.get("doc_language", "en")
        examples = EXAMPLE_QUERIES.get(doc_lang, EXAMPLE_QUERIES["en"])

        st.markdown("**Try an example:**")
        cols = st.columns(len(examples))
        for col, example in zip(cols, examples):
            with col:
                if st.button(
                    example,
                    key=f"example_{hash(example)}",
                    use_container_width=True,
                ):
                    st.session_state.prefilled_query = example

    # ── Query input ────────────────────────────────────────────────────────────
    prefilled = st.session_state.pop("prefilled_query", "")

    query = st.text_area(
        "Your question",
        value=prefilled,
        height=100,
        placeholder=(
            "Ask anything about your document...\n"
            "Hindi: इस दस्तावेज़ में क्या लिखा है?\n"
            "Tamil: இந்த ஆவணத்தில் என்ன உள்ளது?"
        ),
        label_visibility="collapsed",
        key="query_input",
    )

    # ── Submit button ──────────────────────────────────────────────────────────
    ask_col, clear_col = st.columns([4, 1])
    with ask_col:
        submitted = st.button(
            "🔍 Ask SmartDocs",
            use_container_width=True,
            type="primary",
            disabled=not st.session_state.get("doc_id"),
            help="Upload a document first" if not st.session_state.get("doc_id") else "",
        )
    with clear_col:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.pop("last_answer", None)
            st.session_state.pop("last_sources", None)
            st.session_state.pop("last_metadata", None)
            st.rerun()

    if submitted and not query.strip():
        st.warning("Please enter a question before clicking Ask.")
        submitted = False

    return query.strip(), language_override, submitted