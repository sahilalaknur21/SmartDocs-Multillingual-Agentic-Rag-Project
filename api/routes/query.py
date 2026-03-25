# smartdocs/api/routes/query.py
"""
WHY THIS EXISTS:
POST /query/stream — streams the SmartDocs answer as Server-Sent Events.

SSE DESIGN — 4 events in fixed order:
  1. "sources" → sent immediately after run_query completes retrieval
                  The UI can populate the sources panel before showing the answer.
  2. "answer"  → the full answer text (streaming simulation via chunking)
  3. "metadata" → language, cache_hit, crag_triggered, cost_inr, latency_ms
  4. "done"    → signals the stream is complete, UI can stop the spinner

WHY run_query() NOT astream_events():
  astream_events v2 emits dozens of internal LangGraph events (node starts,
  node ends, channel updates). Mapping those to UI-meaningful events requires
  fragile event name matching that breaks when LangGraph updates.
  run_query() gives us the final state — clean, typed, always the same shape.
  We emit 4 predictable SSE events. The UI never breaks on LangGraph upgrades.

TIMEOUT:
  45 seconds. Sarvam-30B timeout is 45s in sarvam_client.py.
  If run_query takes longer than 60s, the route returns a 504-equivalent SSE error.

curl test (paste into terminal, expects a running server):
  curl -N -X POST http://localhost:8000/query/stream \
    -H "Content-Type: application/json" \
    -H "X-User-ID: user_123" \
    -d '{"query": "What is the demand amount in this GST notice?", "doc_id": "<your_doc_id>"}'

Expected output (4 SSE events):
  event: sources
  data: {"sources": [...], "crag_triggered": false}

  event: answer
  data: {"text": "The demand amount mentioned in the GST notice is..."}

  event: metadata
  data: {"language": "English", "cache_hit": false, "cost_inr": 0.5, "latency_ms": 1200}

  event: done
  data: {"status": "complete"}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

# Total timeout for run_query including Sarvam generation
QUERY_TIMEOUT_SECONDS = 60


# ── Request model ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    Request body for POST /query/stream.
    All fields validated by Pydantic before the route handler runs.
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User query in any Indian language or English",
    )
    doc_id: Optional[str] = Field(
        default=None,
        description="Restrict retrieval to specific document UUID. "
                    "If None, searches all user documents.",
    )
    doc_title: str = Field(
        default="your document",
        max_length=200,
        description="Document title for citations in the answer",
    )


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse_event(event_name: str, data: dict) -> str:
    """
    Formats a single SSE event as a string.
    SSE spec: each event is 'event: <name>\ndata: <json>\n\n'
    The double newline (\n\n) is mandatory — it terminates the event.
    """
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_error(message: str, code: str = "QUERY_ERROR") -> str:
    """Formats an SSE error event. UI checks for event: error to show error state."""
    return _sse_event("error", {"error": message, "code": code})


async def _stream_query_events(
    query: str,
    user_id: str,
    doc_id: Optional[str],
    doc_title: str,
) -> AsyncIterator[str]:
    """
    Core SSE generator. Yields 4 events in fixed order.

    Always yields at least one event — even on error.
    The 'done' event is ALWAYS the last event emitted.
    If an error occurs, 'error' event is emitted, then 'done'.
    This guarantees the UI spinner always stops.

    Yields:
        SSE-formatted strings (event: name\ndata: json\n\n)
    """
    start_time = time.perf_counter()

    try:
        # Run the full SmartDocs graph
        # asyncio.wait_for enforces the timeout so the stream doesn't hang
        final_state = await asyncio.wait_for(
            _run_query_safe(query=query, user_id=user_id, doc_id=doc_id, doc_title=doc_title),
            timeout=QUERY_TIMEOUT_SECONDS,
        )

    except asyncio.TimeoutError:
        logger.error(
            f"[query] Timeout after {QUERY_TIMEOUT_SECONDS}s: query={query[:80]} user={user_id}"
        )
        yield _sse_error(
            f"Query timed out after {QUERY_TIMEOUT_SECONDS} seconds. "
            "Try a shorter or more specific question.",
            code="TIMEOUT",
        )
        yield _sse_event("done", {"status": "error"})
        return

    except Exception as e:
        logger.exception(f"[query] Unexpected error: {e} | user={user_id}")
        yield _sse_error(f"Unexpected error: {str(e)[:200]}", code="INTERNAL_ERROR")
        yield _sse_event("done", {"status": "error"})
        return

    # ── Event 1: sources ──────────────────────────────────────────────────────
    # Emitted first — UI populates sources panel immediately.
    # User sees what was retrieved WHILE reading the answer.
    cited_sources = final_state.get("cited_sources", [])
    crag_triggered = final_state.get("crag_triggered", False)

    yield _sse_event("sources", {
        "sources": cited_sources,
        "crag_triggered": crag_triggered,
        "source_count": len(cited_sources),
    })

    # Small yield to flush the sources event to the client before answer
    await asyncio.sleep(0)

    # ── Event 2: answer ───────────────────────────────────────────────────────
    # Full answer text in one event.
    # The Streamlit UI can render this progressively by streaming characters,
    # or display it all at once — the SSE contract doesn't dictate UI behavior.
    final_answer = final_state.get("final_answer", "")

    yield _sse_event("answer", {
        "text": final_answer,
    })

    await asyncio.sleep(0)

    # ── Event 3: metadata ─────────────────────────────────────────────────────
    # Language, cost, cache status — shown in the cost panel (LAW 18).
    lang_result = final_state.get("lang_result")
    cost = final_state.get("cost")
    total_latency_ms = final_state.get("total_latency_ms", 0.0)
    wall_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    yield _sse_event("metadata", {
        "language": lang_result.language_name if lang_result else "Unknown",
        "language_code": lang_result.language_code if lang_result else "en",
        "cache_hit": final_state.get("cache_hit", False),
        "crag_triggered": crag_triggered,
        "cost_inr": round(cost.total_inr, 4) if cost else 0.0,
        "graph_latency_ms": round(total_latency_ms, 2),
        "wall_latency_ms": wall_latency_ms,
        "retry_count": final_state.get("retry_count", 0),
    })

    await asyncio.sleep(0)

    # ── Event 4: done ─────────────────────────────────────────────────────────
    # Always the last event. UI uses this to stop the spinner.
    yield _sse_event("done", {"status": "complete"})

    logger.info(
        f"[query] ✓ Stream complete: user={user_id} "
        f"lang={lang_result.language_code if lang_result else '?'} "
        f"cache={final_state.get('cache_hit', False)} "
        f"latency={wall_latency_ms}ms"
    )


async def _run_query_safe(
    query: str,
    user_id: str,
    doc_id: Optional[str],
    doc_title: str,
) -> dict:
    """
    Thin wrapper around run_query.
    Separated from the generator so exceptions propagate cleanly
    to the try/except in _stream_query_events.
    """
    from agents.smartdocs_graph import run_query
    return await run_query(
        query=query,
        user_id=user_id,
        doc_id=doc_id,
        doc_title=doc_title,
    )


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/query/stream")
async def query_stream(body: QueryRequest, request: Request) -> StreamingResponse:
    """
    Streams SmartDocs answer as Server-Sent Events.

    Returns 4 events in order:
      sources → answer → metadata → done

    The 'done' event is always emitted — even on error.
    Check event type 'error' for failures.

    curl test:
      curl -N -X POST http://localhost:8000/query/stream \
        -H "Content-Type: application/json" \
        -H "X-User-ID: user_123" \
        -d '{"query": "What is the GST demand amount?", "doc_id": "your-uuid"}'
    """
    user_id: str = request.state.user_id

    logger.info(
        f"[query] Request: user={user_id} "
        f"query={body.query[:80]!r} doc_id={body.doc_id}"
    )

    return StreamingResponse(
        _stream_query_events(
            query=body.query,
            user_id=user_id,
            doc_id=body.doc_id,
            doc_title=body.doc_title,
        ),
        media_type="text/event-stream",
        headers={
            # Prevent proxy buffering — critical for SSE to work through nginx/Railway
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            # CORS header for browser SSE requests
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.post("/query")
async def query_sync(body: QueryRequest, request: Request) -> dict:
    """
    Non-streaming query endpoint — returns complete response as JSON.
    Useful for testing without SSE client, and for evaluation scripts.

    curl test:
      curl -X POST http://localhost:8000/query \
        -H "Content-Type: application/json" \
        -H "X-User-ID: user_123" \
        -d '{"query": "What is the GST demand amount?", "doc_id": "your-uuid"}'
    """
    user_id: str = request.state.user_id

    try:
        final_state = await asyncio.wait_for(
            _run_query_safe(
                query=body.query,
                user_id=user_id,
                doc_id=body.doc_id,
                doc_title=body.doc_title,
            ),
            timeout=QUERY_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Query timed out after {QUERY_TIMEOUT_SECONDS} seconds",
        )
    except Exception as e:
        logger.exception(f"[query] Sync query error: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:200])

    lang_result = final_state.get("lang_result")
    cost = final_state.get("cost")

    return {
        "answer": final_state.get("final_answer", ""),
        "sources": final_state.get("cited_sources", []),
        "language": lang_result.language_name if lang_result else "Unknown",
        "language_code": lang_result.language_code if lang_result else "en",
        "cache_hit": final_state.get("cache_hit", False),
        "crag_triggered": final_state.get("crag_triggered", False),
        "cost_inr": round(cost.total_inr, 4) if cost else 0.0,
        "latency_ms": round(final_state.get("total_latency_ms", 0.0), 2),
        "retry_count": final_state.get("retry_count", 0),
    }