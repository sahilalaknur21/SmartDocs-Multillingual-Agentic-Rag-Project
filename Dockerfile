# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.13-slim-bookworm AS builder

# System packages required for ML + PDF + Indic NLP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install uv — fast Python package manager
COPY --from=ghcr.io/astral-sh/uv:0.5.17 /uv /uvx /usr/local/bin/

WORKDIR /build

# Copy only dependency manifest — cache this layer
COPY pyproject.toml ./

# Install all dependencies into /build/.venv
RUN uv venv .venv && \
    uv pip install --python .venv/bin/python \
        fastapi uvicorn[standard] python-multipart \
        langchain-core langgraph langsmith openai \
        langchain-openai langchain-huggingface \
        sentence-transformers rank-bm25 scikit-learn \
        langdetect pdfplumber asyncpg pgvector \
        flashrank redis pydantic pydantic-settings \
        python-dotenv pyyaml aiohttp tavily-python \
        indic-nlp-library presidio-analyzer presidio-anonymizer \
        opentelemetry-sdk tenacity httpx numpy

# Download indic_nlp_resources — required by indic-nlp-library at runtime
RUN git clone --depth 1 \
    https://github.com/anoopkunchukuttan/indic_nlp_resources.git \
    /indic_nlp_resources

# ── Stage 2: production API image ─────────────────────────────────────────────
FROM python:3.13-slim-bookworm AS api

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — never run production as root
RUN useradd -m -u 1000 -s /bin/bash appuser

WORKDIR /app

# Copy installed venv from builder
COPY --from=builder /build/.venv /app/.venv

# Copy indic resources
COPY --from=builder /indic_nlp_resources /app/indic_nlp_resources

# Copy application source
COPY --chown=appuser:appuser . .

# Environment — production defaults
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INDIC_RESOURCES_PATH="/app/indic_nlp_resources" \
    EMBEDDING_DEVICE="cpu" \
    ENVIRONMENT="production" \
    SENTENCE_TRANSFORMERS_HOME="/app/.cache/sentence_transformers" \
    HF_HOME="/app/.cache/huggingface" \
    PORT=8000

RUN mkdir -p /app/.cache/sentence_transformers /app/.cache/huggingface && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ping')" || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --log-level info"]