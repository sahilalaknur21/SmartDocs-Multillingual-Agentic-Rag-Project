# ── SmartDocs API — Hugging Face Spaces Docker ────────────────────────────────
# HF Spaces requirement: expose port 7860, run as non-root user 1000
# CPU Basic: 2 vCPU, 16GB RAM, 50GB storage — no image size restrictions

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.13-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.17 /uv /uvx /usr/local/bin/

WORKDIR /build

# Install all production dependencies
RUN uv venv .venv && \
    uv pip install --python .venv/bin/python \
        fastapi "uvicorn[standard]" python-multipart \
        langchain-core langgraph langsmith openai \
        langchain-openai langchain-huggingface \
        sentence-transformers rank-bm25 scikit-learn \
        langdetect pdfplumber asyncpg pgvector \
        flashrank redis pydantic "pydantic-settings" \
        python-dotenv pyyaml aiohttp tavily-python \
        indic-nlp-library presidio-analyzer presidio-anonymizer \
        "opentelemetry-sdk" tenacity httpx numpy

# Clone indic_nlp_resources — required for Indic sentence tokenization
RUN git clone --depth 1 \
    https://github.com/anoopkunchukuttan/indic_nlp_resources.git \
    /indic_nlp_resources

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces REQUIRES user 1000
RUN useradd -m -u 1000 -s /bin/bash appuser

WORKDIR /app

COPY --from=builder /build/.venv /app/.venv
COPY --from=builder /indic_nlp_resources /app/indic_nlp_resources

# Copy application source
COPY --chown=appuser:appuser . .

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INDIC_RESOURCES_PATH="/app/indic_nlp_resources" \
    EMBEDDING_DEVICE="cpu" \
    EMBEDDING_BATCH_SIZE="8" \
    ENVIRONMENT="production" \
    SENTENCE_TRANSFORMERS_HOME="/app/.cache/st" \
    HF_HOME="/app/.cache/hf" \
    PORT=7860

RUN mkdir -p /app/.cache/st /app/.cache/hf && \
    chown -R appuser:appuser /app

USER appuser

# HF Spaces uses port 7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health/ping')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--log-level", "info"]