# smartdocs/api/routes/health.py
"""
WHY THIS EXISTS:
/health is the first endpoint curl hits after deploy.
It checks every critical dependency: DB, Redis, embedding model, graph.
Returns structured JSON so Railway/monitoring can parse it.

curl test:
  curl http://localhost:8000/health
  Expected: {"status": "ok", "checks": {"db": "ok", "redis": "ok", ...}}

If any check is "error", status is "degraded" — not "ok".
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()


async def _check_database() -> tuple[str, str]:
    """Verifies pgvector pool is open and accepts a query."""
    try:
        from vectorstore.pgvector_client import get_pgvector_client
        client = get_pgvector_client()
        if client._pool is None:
            return "error", "pool not initialized"
        async with client._pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
        return "ok", f"query returned {result}"
    except Exception as e:
        return "error", str(e)[:120]


async def _check_redis() -> tuple[str, str]:
    """Verifies Redis connection accepts a ping."""
    try:
        import redis.asyncio as aioredis
        from config.settings import get_settings
        settings = get_settings()
        redis = await aioredis.from_url(settings.redis_url, socket_connect_timeout=2)
        pong = await redis.ping()
        await redis.aclose()
        return "ok", f"PING → {pong}"
    except Exception as e:
        return "degraded", f"Redis unavailable: {str(e)[:80]}"


async def _check_embedding_model() -> tuple[str, str]:
    """Verifies DenseEmbedder singleton is loaded (model in VRAM/RAM)."""
    try:
        from embeddings.dense_embedder import get_embedder
        embedder = get_embedder()
        # Model loaded if _model is not None — no inference call needed
        if embedder._model is not None:
            return "ok", f"model loaded on {embedder.device}"
        else:
            return "degraded", "model not yet loaded (will load on first query)"
    except Exception as e:
        return "error", str(e)[:120]


async def _check_graph() -> tuple[str, str]:
    """Verifies LangGraph compiled correctly."""
    try:
        from agents.smartdocs_graph import get_smartdocs_graph
        graph = get_smartdocs_graph()
        return "ok", f"graph compiled: {type(graph).__name__}"
    except Exception as e:
        return "error", str(e)[:120]


@router.get("/health")
async def health_check() -> JSONResponse:
    """
    Full health check.
    Runs all dependency checks in sequence.
    Returns 200 if all ok/degraded, 503 if any are error.

    curl test:
      curl -s http://localhost:8000/health | python -m json.tool
    """
    start = time.perf_counter()

    db_status, db_detail = await _check_database()
    redis_status, redis_detail = await _check_redis()
    embed_status, embed_detail = await _check_embedding_model()
    graph_status, graph_detail = await _check_graph()

    checks: dict[str, Any] = {
        "db":        {"status": db_status,    "detail": db_detail},
        "redis":     {"status": redis_status,  "detail": redis_detail},
        "embedding": {"status": embed_status,  "detail": embed_detail},
        "graph":     {"status": graph_status,  "detail": graph_detail},
    }

    # Overall status: ok if all ok, degraded if any degraded, error if any error
    statuses = [db_status, redis_status, embed_status, graph_status]
    if "error" in statuses:
        overall = "error"
        http_status = 503
    elif "degraded" in statuses:
        overall = "degraded"
        http_status = 200
    else:
        overall = "ok"
        http_status = 200

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return JSONResponse(
        status_code=http_status,
        content={
            "status": overall,
            "checks": checks,
            "latency_ms": latency_ms,
        },
    )


@router.get("/health/ping")
async def ping() -> dict:
    """
    Lightweight liveness probe for Railway health checks.
    Does NOT check dependencies — just confirms server is alive.

    curl test:
      curl http://localhost:8000/health/ping
      Expected: {"status": "alive"}
    """
    return {"status": "alive"}
