# smartdocs/ui/components/answer_panel.py
"""
WHY THIS EXISTS:
Consumes the SSE stream from POST /query/stream and renders:
  1. Sources panel — populated immediately when 'sources' event arrives
  2. Answer — streamed character by character as tokens arrive
  3. Language badge — shown in answer header
  4. CRAG disclosure — when answer comes from web search
 
SSE CONSUMER DESIGN:
  Streamlit is synchronous. We use httpx.stream() in sync mode.
  st.empty() placeholders update in-place — no page rerender.
  Each SSE event updates a specific placeholder.
 
  Event order: sources → answer → metadata → done
  Sources update BEFORE the answer renders — LAW 9 compliance.
 
THINK TAG STRIPPING:
  Sarvam-M exposes chain-of-thought in <think>...</think> tags.
  These are stripped client-side before rendering.
  The reasoning is real but showing it confuses non-technical users.
"""
 
from __future__ import annotations
 
import json
import re
import time
 
import httpx
import streamlit as st
 
 
def _strip_think_tags(text: str) -> str:
    """
    Strips Sarvam-M chain-of-thought reasoning blocks.
    <think>...</think> content is valid reasoning — not shown to users.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
 
 
def _render_sources(sources: list[dict], crag_triggered: bool) -> None:
    """Renders cited sources in an expandable panel."""
    if not sources:
        return
 
    label = f"📚 Sources ({len(sources)})"
    if crag_triggered:
        label += " · includes web search"
 
    with st.expander(label, expanded=True):
        for i, source in enumerate(sources, 1):
            is_web = source.get("is_crag", False)
            icon = "🌐" if is_web else "📄"
            title = source.get("title", "Document")
            page = source.get("page_number", 0)
            score = source.get("reranker_score", 0.0)
 
            if is_web:
                url = source.get("url", "")
                st.markdown(
                    f"{icon} **{title}** · [web]({url}) · relevance: `{score:.2f}`"
                    if url
                    else f"{icon} **{title}** · web search · relevance: `{score:.2f}`"
                )
            else:
                st.markdown(
                    f"{icon} **{title}** · page {page} · relevance: `{score:.2f}`"
                )
 
            if i < len(sources):
                st.divider()
 
 
def _render_answer_text(text: str, language: str, language_code: str) -> None:
    """Renders the final cleaned answer with language indicator."""
    lang_flags = {
        "hi": "🇮🇳", "ta": "🇮🇳", "te": "🇮🇳", "kn": "🇮🇳",
        "ml": "🇮🇳", "mr": "🇮🇳", "en": "🇬🇧", "gu": "🇮🇳",
        "pa": "🇮🇳", "bn": "🇮🇳", "ur": "🇮🇳",
    }
    flag = lang_flags.get(language_code, "🌐")
 
    st.markdown(
        f"<div style='color: #888; font-size: 0.8em; margin-bottom: 8px;'>"
        f"{flag} Answered in {language}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(text)
 
 
def stream_and_render_answer(
    query: str,
    doc_id: str,
    doc_title: str,
    api_base: str,
    user_id: str,
) -> None:
    """
    Calls POST /query/stream, consumes 4 SSE events, renders progressively.
 
    Rendering order (matches SSE event order):
      1. Spinner shown immediately
      2. 'sources' event → sources panel renders, spinner text updates
      3. 'answer' event → answer renders (think tags stripped)
      4. 'metadata' event → stored in session_state for cost_panel
      5. 'done' event → spinner dismissed
 
    On any error: error event or exception → renders error message, never hangs.
    """
    # Placeholders — updated in-place as events arrive
    spinner_placeholder = st.empty()
    sources_placeholder = st.empty()
    answer_placeholder = st.empty()
 
    spinner_placeholder.info("🔍 Searching your document...")
 
    collected_sources = []
    collected_answer = ""
    collected_metadata = {}
    crag_triggered = False
    error_occurred = False
 
    try:
        with httpx.stream(
            "POST",
            f"{api_base}/query/stream",
            headers={
                "X-User-ID": user_id,
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "doc_id": doc_id,
                "doc_title": doc_title,
            },
            timeout=120.0,
        ) as response:
 
            if response.status_code != 200:
                spinner_placeholder.empty()
                st.error(f"Query failed: HTTP {response.status_code}")
                return
 
            current_event = None
 
            for line in response.iter_lines():
                line = line.strip()
                if not line:
                    continue
 
                if line.startswith("event:"):
                    current_event = line[len("event:"):].strip()
                    continue
 
                if line.startswith("data:") and current_event:
                    raw_data = line[len("data:"):].strip()
 
                    try:
                        data = json.loads(raw_data)
                    except json.JSONDecodeError:
                        continue
 
                    # ── sources event ─────────────────────────────────────────
                    if current_event == "sources":
                        collected_sources = data.get("sources", [])
                        crag_triggered = data.get("crag_triggered", False)
                        spinner_placeholder.info("✍️ Generating answer...")
                        with sources_placeholder.container():
                            _render_sources(collected_sources, crag_triggered)
 
                    # ── answer event ──────────────────────────────────────────
                    elif current_event == "answer":
                        raw_answer = data.get("text", "")
                        collected_answer = _strip_think_tags(raw_answer)
                        spinner_placeholder.empty()
                        with answer_placeholder.container():
                            _render_answer_text(
                                collected_answer,
                                language=collected_metadata.get("language", "English"),
                                language_code=collected_metadata.get("language_code", "en"),
                            )
 
                    # ── metadata event ────────────────────────────────────────
                    elif current_event == "metadata":
                        collected_metadata = data
                        # Re-render answer with language info now available
                        if collected_answer:
                            with answer_placeholder.container():
                                _render_answer_text(
                                    collected_answer,
                                    language=data.get("language", "English"),
                                    language_code=data.get("language_code", "en"),
                                )
 
                    # ── error event ───────────────────────────────────────────
                    elif current_event == "error":
                        error_occurred = True
                        spinner_placeholder.empty()
                        error_msg = data.get("error", "Unknown error")
                        error_code = data.get("code", "")
                        answer_placeholder.error(f"❌ {error_msg} ({error_code})")
 
                    # ── done event ────────────────────────────────────────────
                    elif current_event == "done":
                        spinner_placeholder.empty()
                        break
 
                    current_event = None
 
    except httpx.TimeoutException:
        spinner_placeholder.empty()
        answer_placeholder.error(
            "⏱️ Query timed out after 120 seconds. "
            "Try a shorter or more specific question."
        )
        error_occurred = True
 
    except httpx.ConnectError:
        spinner_placeholder.empty()
        answer_placeholder.error(
            "🔌 Cannot connect to SmartDocs API. "
            f"Is the server running at {api_base}?"
        )
        error_occurred = True
 
    except Exception as e:
        spinner_placeholder.empty()
        answer_placeholder.error(f"❌ Unexpected error: {str(e)[:200]}")
        error_occurred = True
 
    # ── Persist results in session_state ──────────────────────────────────────
    if not error_occurred and collected_answer:
        st.session_state.last_answer = collected_answer
        st.session_state.last_sources = collected_sources
        st.session_state.last_metadata = collected_metadata
        st.session_state.last_crag = crag_triggered
 
        # Accumulate session cost
        cost_this_query = collected_metadata.get("cost_inr", 0.0)
        st.session_state.session_cost_inr = (
            st.session_state.get("session_cost_inr", 0.0) + cost_this_query
        )
        st.session_state.query_count = st.session_state.get("query_count", 0) + 1
 
 
def render_previous_answer() -> None:
    """
    Re-renders the last answer from session_state without re-querying.
    Called on Streamlit reruns (e.g. after clicking a button) to preserve
    the answer panel state.
    """
    if not st.session_state.get("last_answer"):
        return
 
    sources = st.session_state.get("last_sources", [])
    crag = st.session_state.get("last_crag", False)
    meta = st.session_state.get("last_metadata", {})
    answer = st.session_state.get("last_answer", "")
 
    _render_sources(sources, crag)
    _render_answer_text(
        answer,
        language=meta.get("language", "English"),
        language_code=meta.get("language_code", "en"),
    )