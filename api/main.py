# smartdocs/api/main.py
"""
WHY THIS EXISTS:
FastAPI application entry point.
Handles startup (DB connect, model warmup), shutdown (pool close),
middleware registration (user context), and route mounting.

LIFESPAN:
  Uses the modern FastAPI lifespan context manager (not deprecated @app.on_event).
  Startup: connects pgvector pool, warms up DenseEmbedder.
  Shutdown: closes pgvector pool cleanly.

  WHY WARMUP ON STARTUP:
    multilingual-e5-large loads in 3-5s on first call.
    Without warmup, the first user query experiences a 5s delay.
    Warmup moves that cost to server startup where it's invisible to users.

HOW TO RUN:
  From: smartdocs/
  Dev:  uv run uvicorn api.main:app --reload --port 8000 --log-level info
  Prod: uv run uvicorn api.main:app --port 8000 --workers 1

  NOTE: --workers 1 in production.
  DenseEmbedder holds model in VRAM. Multiple workers = multiple model loads
  = OOM on RTX 3050 6GB. One worker, async concurrency handles the load.

VERIFY SERVER IS RUNNING:
  curl http://localhost:8000/health/ping
  Expected: {"status": "alive"}

  curl http://localhost:8000/health
  Expected: {"status": "ok", "checks": {...}}
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware.user_context import UserContextMiddleware
from api.routes import health, ingest, query
from config.settings import get_settings

settings = get_settings()

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup and shutdown.
    FastAPI recommends this over @app.on_event (deprecated in 0.95+).

    Startup sequence:
      1. Connect pgvector pool (fail fast if DB unreachable)
      2. Warm up DenseEmbedder (loads multilingual-e5-large into VRAM)
      3. Compile LangGraph (validates all nodes and edges)
      → Server ready to accept requests

    Shutdown sequence:
      1. Close pgvector pool (drains in-flight queries)
    """
    # ── STARTUP ───────────────────────────────────────────────────────────────
    logger.info("SmartDocs API starting up...")

    # 1. Connect pgvector pool
    try:
        from vectorstore.pgvector_client import get_pgvector_client
        pgclient = get_pgvector_client()
        await pgclient.connect()
        logger.info("✓ pgvector pool connected")
    except Exception as e:
        logger.error(f"✗ pgvector connection FAILED: {e}")
        logger.error("Check DATABASE_URL in .env — server will start but DB queries will fail")
        # Don't raise — server starts degraded, /health will report the error

    # 2. Warm up embedding model
    try:
        from embeddings.dense_embedder import get_embedder
        import asyncio
        embedder = get_embedder()
        loop = asyncio.get_running_loop()
        # embed_passages is synchronous — run in executor to avoid blocking event loop
        await loop.run_in_executor(None, embedder.embed_passages, ["warmup"])
        logger.info(f"✓ DenseEmbedder warmed up on {embedder.device}")
    except Exception as e:
        logger.warning(f"⚠ DenseEmbedder warmup failed: {e} — will load on first query")

    # 3. Compile LangGraph (validates all nodes and edges)
    try:
        from agents.smartdocs_graph import get_smartdocs_graph
        get_smartdocs_graph()
        logger.info("✓ LangGraph compiled")
    except Exception as e:
        logger.error(f"✗ LangGraph compile FAILED: {e}")

    logger.info(
        f"SmartDocs API ready | "
        f"env={settings.environment} | "
        f"model={settings.sarvam_model_30b}"
    )

    yield  # Server runs here

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info("SmartDocs API shutting down...")

    try:
        from vectorstore.pgvector_client import get_pgvector_client
        pgclient = get_pgvector_client()
        await pgclient.close()
        logger.info("✓ pgvector pool closed")
    except Exception as e:
        logger.warning(f"pgvector pool close error: {e}")

    logger.info("SmartDocs API shutdown complete")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SmartDocs API",
    description=(
        "Multilingual PDF Q&A for Indian professionals. "
        "22 Indian languages. No translation layer. "
        "Powered by Sarvam-30B + multilingual-e5-large."
    ),
    version="1.0.0",
    lifespan=lifespan,
    # Disable automatic redirect: /ingest → /ingest/
    # Redirect strips headers including X-User-ID — breaks auth middleware
    redirect_slashes=False,
)

# ── Middleware ────────────────────────────────────────────────────────────────
# Order matters: middleware executes in REVERSE registration order.
# UserContextMiddleware must run after CORS — register CORS last.

# User context — injects user_id into request.state for every request
app.add_middleware(UserContextMiddleware)

# CORS — allow all origins in development
# In production: replace "*" with your actual frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else [
        "https://your-streamlit-app.railway.app",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(health.router, tags=["Health"])
app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(query.router, tags=["Query"])


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/")
async def root() -> dict:
    """
    Root endpoint — confirms API is running.
    curl http://localhost:8000/
    """
    return {
        "name": "SmartDocs API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }