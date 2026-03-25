# smartdocs/ui/components/cost_panel.py
"""
WHY THIS EXISTS:
Displays running cost per query and session total in INR. LAW 18.
Solo operator survival depends on knowing cost before scaling.
multilingual-e5-large = ₹0.00 (local CPU/GPU).
Sarvam-30B = per-token cost tracked from API response.

Shown in sidebar below document info.
Updates after every query — not just at session end.
"""

from __future__ import annotations

import streamlit as st


def render_cost_panel() -> None:
    """
    Renders the cost tracking panel in the sidebar.
    Reads from session_state — no API call needed.

    State read:
      - session_cost_inr: float — total cost this session
      - query_count: int — number of queries this session
      - last_metadata: dict — metadata from last query (cost, latency, cache)
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💰 Cost Tracker")

    last_meta = st.session_state.get("last_metadata", {})
    session_cost = st.session_state.get("session_cost_inr", 0.0)
    query_count = st.session_state.get("query_count", 0)

    # ── Per-query cost ─────────────────────────────────────────────────────────
    if last_meta:
        last_cost = last_meta.get("cost_inr", 0.0)
        cache_hit = last_meta.get("cache_hit", False)
        latency = last_meta.get("wall_latency_ms", 0.0)
        crag = last_meta.get("crag_triggered", False)

        if cache_hit:
            st.sidebar.success("⚡ Cache hit — ₹0.00")
        else:
            cost_display = f"₹{last_cost:.4f}" if last_cost > 0 else "₹0.00"
            st.sidebar.metric(
                label="Last query cost",
                value=cost_display,
                help="Sarvam-30B generation cost in INR",
            )

        # Latency
        latency_sec = latency / 1000
        if latency_sec < 5:
            latency_color = "🟢"
        elif latency_sec < 15:
            latency_color = "🟡"
        else:
            latency_color = "🔴"

        st.sidebar.caption(
            f"{latency_color} {latency_sec:.1f}s latency"
            + (" · CRAG" if crag else "")
            + (" · cached" if cache_hit else "")
        )

    # ── Session totals ─────────────────────────────────────────────────────────
    if query_count > 0:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric(
                label="Session total",
                value=f"₹{session_cost:.4f}",
            )
        with col2:
            st.metric(
                label="Queries",
                value=str(query_count),
            )

        avg = session_cost / query_count if query_count > 0 else 0.0
        st.sidebar.caption(f"Avg ₹{avg:.4f} / query")

    # ── Embedding cost note ────────────────────────────────────────────────────
    st.sidebar.markdown(
        """
        <div style="font-size:0.75em; color:#888; margin-top:8px;">
        🖥️ Embeddings: ₹0.00 (local CPU)<br>
        ☁️ Generation: Sarvam-30B API
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Reset button ───────────────────────────────────────────────────────────
    if query_count > 0:
        if st.sidebar.button("Reset cost counter", use_container_width=True):
            st.session_state.session_cost_inr = 0.0
            st.session_state.query_count = 0
            st.rerun()
